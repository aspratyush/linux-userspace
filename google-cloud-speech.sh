API_KEY='AIzaSyBwFmH8Axxyg_Z7IgCexqeJlUQxi8BKhhw'

# Create a request file with our JSON request in the current directory.
FILENAME="request-"`date +"%s".json`
cat <<EOF > $FILENAME
{
  "config": {
    "encoding":"FLAC",
    "sampleRate":16000,
    "maxAlternatives": 3,
    "languageCode": "en-US",
    "speechContext": {
      "phrases": [
        "DevFest"
      ]
    }
  },
  "audio": {
    "content":
  }
}
EOF

# Update the languageCode parameter if one was supplied
if [ $# -eq 1 ]
  then
    sed -i '' -e "s/en-US/$2/g" $FILENAME
fi

# Record an audio file, base64 encode it, and update our request object
#read -p "Press enter when you're ready to record" rec
#if [ -z $rec ]; then
#  rec --channels=1 --bits=16 --rate=16000 audio.flac trim 0 5
#  echo \"`base64 audio.flac`\" > audio.base64
#  sed -i '' -e '/"content":/r audio.base64' $FILENAME
#fi
base64 $1 > audio.base64
sed -i '' -e '/"content":/r audio.base64' $FILENAME
echo Request "file" $FILENAME created:

head -8 $FILENAME # Don't print the entire file because there's a giant base64 string
echo $'\t"Your base64 string..."\n\x20\x20}\n}'
curl -s -X POST -H "Content-Type: application/json" --data-binary @${FILENAME} https://speech.googleapis.com/v1/speech:recognize?key=$API_KEY

# Call the speech API
#read -p $'\nPress enter when you\'re ready to call the Speech API' var
#if [ -z $var ];
#  then
#    echo "Running the following curl command:"
#    # echo "curl -s -X POST -H 'Content-Type: application/json' --data-binary @${FILENAME} https://speech.googleapis.com/v1beta1/speech:syncrecognize?key=API_KEY"
#    curl -s -X POST -H "Content-Type: application/json" --data-binary @${FILENAME} https://speech.googleapis.com/v1/speech:recognize?key=$API_KEY
#fi
