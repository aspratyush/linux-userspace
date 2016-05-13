ALSA
=========

# Record audio
`$ arecord -t raw -c 1 -f S16_LE -r 16000 recording.raw`

Options:
* -t : extension (raw / wav)
* -c : channels (1/2)
* -f : encoding format
* -r : sampling rate (in Hz)

# Play audio
`$ aplay -t raw -c 1 -r 16000 -f S16_LE recording.raw`

Options:
* -t : extension (raw / wav)
* -c : channels (1/2)
* -f : encoding format
* -r : sampling rate (in Hz)
