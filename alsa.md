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


# Using Google Speech to Text WebAPI through curl
`$ curl -X POST --max-time 15 --data-binary @'<path_to_file>' --header 'Content-Type: audio/l16; rate=16000;' 'https://www.google.com/speech-api/v2/recognize?output=json&lang=en-us&key=<key>&client=chromium&maxresults=1&pfilter=2'`
