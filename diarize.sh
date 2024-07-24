# For all files in the directory, run the diarization script
# Usage: ./diarize.sh <path-to-directory>
# Example: ./diarize.sh /home/username/audio-files

# Check if the user has provided a directory
if [ $# -eq 0 ]; then
    echo "Please provide the path to the directory containing the audio files"
    exit 1
fi

# Check if the directory exists
if [ ! -d $1 ]; then
    echo "Directory does not exist"
    exit 1
fi

# Loop through all the files in the directory
for file in $1/*; do
    # Check if the file is an audio file
    if [ -f $file ]; then
        # Run the diarization script
        python diarize.py $file
    fi
done

# End of script
```
