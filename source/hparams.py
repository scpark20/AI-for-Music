from easydict import EasyDict

dims = EasyDict(interval = 100,
                velocity = 32,
                note_on = 128,
                note_off = 128,
                pedal_on = 1,
                pedal_off = 1)

offsets = EasyDict(interval = 100,
                   velocity = dims.interval,
                   note_on = dims.interval + dims.velocity,
                   note_off = dims.interval + dims.velocity + dims.note_on,
                   pedal_on = dims.interval + dims.velocity + dims.note_on + dims.note_off,
                   pedal_off = dims.interval + dims.velocity + dims.note_on + dims.note_off + dims.pedal_on)

dataset_hparams = EasyDict(root_dir = 'dataset/',
                           max_note_duration = 2, # seconds)
                           token_length = 500,
                           dims = dims,
                           offsets = offsets
                          )

model_hparams = EasyDict(n_tokens = dataset_hparams.offsets.pedal_off + 1,
                         embedding_dim = 512,
                         hidden_dim = 1024
                        )      