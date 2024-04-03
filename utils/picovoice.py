import pvleopard

handle = pvleopard.create(access_key='pwstl2QADUyrDS3tlGAH22iH+0JcpX0TQx2gUljFgLZeBoxPRHfHvw==')

audio_file = "/fluent_speech_commands_dataset/wavs/speakers/2ojo7YRL7Gck83Z3/0d4b72b0-45e0-11e9-81ce-69b74fd7e64e.wav"

text = handle.process_file(audio_file)
print(handle.process_file(audio_file))