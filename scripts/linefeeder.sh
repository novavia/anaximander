cat words.txt | while read line; do gcloud pubsub topics publish projects/anaximander-tests/topics/word-in --message "$line"; done
