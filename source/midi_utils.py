import mido
import numpy as np

def get_event_list(data_file):
    
    midi_messages = mido.MidiFile(data_file)
    
    current_time = 0
    event_list = []
    pedal_on = False
    
    for message in midi_messages:
        # Accumulate relative times
        current_time += message.time
        
        # 'note_on' 이벤트 발생하고 velocity가 0보다 큰 경우
        if message.type == 'note_on' and message.velocity > 0:
            event = {'time': current_time,
                     'type': 'note_on',
                     'note': message.note,
                     'velocity': message.velocity}
            event_list.append(event)
        
        # 'note_off' 이벤트 발생하거나, 'note_on' 이벤트이면서 velocity가 0인 경우
        # midi 파일에 따라서 note_off 이벤트 대신 note_on, velocity 0으로 표시하는 기록하는 것들이 있음
        elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
            event = {'time': current_time,
                     'type': 'note_off',
                     'note': message.note}
            event_list.append(event)
        
        # Sustain pedal 이벤트 발생한 경우
        elif message.type == 'control_change' and message.control == 64:
            if pedal_on == False and message.value > 0:
                pedal_on = True
                event = {'time': current_time,
                         'type': 'pedal_on'}
                event_list.append(event)
                
            elif pedal_on == True and message.value == 0:
                pedal_on = False
                event = {'time': current_time,
                         'type': 'pedal_off'}
                event_list.append(event)    
    return event_list

def event_list_to_tokens(event_list, hp):
    # token들을 모아줄 list를 만듭니다.
    tokens = []
    prev_time = 0
    
    # event_list에서 event들을 하나씩 꺼내옵니다.
    for event in event_list:
        
        # interval은 현재 event['time']에서 prev_time을 빼서 구합니다.
        interval = event['time'] - prev_time
        
        # interval(시간)을 interval_token(자연수) 값으로 변환합니다.
        interval_token = int(interval / hp.max_note_duration * hp.dims.interval)
        interval_token = min(interval_token, hp.dims.interval)
        
        # tokens에 interval_token을 저장합니다.
        tokens.append(interval_token)
        
        # 현재 event['time']을 prev_time으로 저장합니다.
        prev_time = event['time']
        
        # 이벤트 'note_on'의 경우 velocity token과 note_on token을 저장합니다.
        if event['type'] == 'note_on':
            tokens.append(hp.offsets.velocity + int(event['velocity'] / 128 * hp.dims.velocity))
            tokens.append(hp.offsets.note_on + event['note'])
            
        # 이벤트 'note_off'에 대한 token을 저장합니다.
        elif event['type'] == 'note_off':
            tokens.append(hp.offsets.note_off + event['note'])
            
        # 이벤트 'pedal_on'에 대한 token을 저장합니다.
        elif event['type'] == 'pedal_on':
            tokens.append(hp.offsets.pedal_on)
            
        # 이벤트 'pedal_off'에 대한 token을 저장합니다.
        elif event['type'] == 'pedal_off':
            tokens.append(hp.offsets.pedal_off)
            
    # token sequence를 반환합니다.
    return np.array(tokens)
            
def tokens_to_event_list(tokens, hp):
    current_time = 0
    current_velocity = 0
    event_list = []
    for token in tokens:
        # interval
        if token < hp.offsets.velocity:
            current_time += token / hp.dims.interval * hp.max_note_duration
            
        # velocity
        elif token < hp.offsets.note_on:
            current_velocity = (token - hp.offsets.velocity) / hp.dims.velocity * 128
            
        # note_on
        elif token < hp.offsets.note_off:
            event = {'time': current_time,
                     'type': 'note_on',
                     'note': token - hp.offsets.note_on,
                     'velocity': int(current_velocity)}
            event_list.append(event)
                     
        # note_off
        elif token < hp.offsets.pedal_on:
            event = {'time': current_time,
                     'type': 'note_off',
                     'note': token - hp.offsets.note_off}
            event_list.append(event)
            
        # pedal_on
        elif token < hp.offsets.pedal_off:
            event = {'time': current_time,
                     'type': 'pedal_on'}
            event_list.append(event)
            
        # pedal_off
        else:
            event = {'time': current_time,
                     'type': 'pedal_off'}
            event_list.append(event)
    
    return event_list

def save_event_list_to_midi_file(event_list, midi_file, speed=1.0):
    '''
    MIDI Library for Saving
    https://github.com/louisabraham/python3-midi
    '''
    import midi
    
    RESOULUTION = 100
    BPM = 60
    def second_to_tick(second, resolution=RESOULUTION, bpm=BPM):
        return int(second * resolution * (bpm/60) * speed)

    # Init. a track.
    pattern = midi.Pattern(resolution=RESOULUTION)
    track = midi.Track()
    pattern.append(track)
    tempo = midi.SetTempoEvent(tick=0, bpm=BPM)
    track.append(tempo)

    # Event loop
    prev_time = 0
    for event in event_list:
        interval = event['time'] - prev_time
        prev_time = event['time']
        
        if event['type'] == 'note_on':
            track.append(midi.NoteOnEvent(tick=second_to_tick(interval), velocity=event['velocity'], pitch=event['note']))
        elif event['type'] == 'note_off':
            track.append(midi.NoteOffEvent(tick=second_to_tick(interval), pitch=event['note']))
        elif event['type'] == 'pedal_on':
            track.append(midi.ControlChangeEvent(tick=second_to_tick(interval), control=64, value=64))
        elif event['type'] == 'pedal_off':
            track.append(midi.ControlChangeEvent(tick=second_to_tick(interval), control=64, value=0))
                         
    # End of Event
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    
    # Save midi file
    midi.write_midifile(midi_file, pattern)
