# Based on https://github.com/ros-drivers/audio_common/blob/main/audio_common_msgs/msg/AudioInfo.msg

std_msgs/Header header

# Chunk size [samples]
uint32 chunk_size
# Number of channels
uint8 channels
# Sampling rate [Hz]
uint32 sample_rate
# Audio encoding (e.g. S16LE)
string encoding
# Audio format
uint8 is_bigendian
# Amount of audio data per second [bits/s]
uint32 bitrate
# Audio coding format (e.g. WAVE, MP3)
string coding_format

uint32 step
uint8[] data