# UG-TTS
UG-TTS is an open-source Text-to-Speech system for Uyghur, offering high-quality voice synthesis for Uyghur language.



# 架构

```c++

+---------------------------+
|    Text Input (文本输入)    |
+---------------------------+
           |
           v
+---------------------------+
| Text Preprocessing (文本预处理)|
+---------------------------+
           |
           v
+---------------------------+
| Phoneme Conversion (音素转换)|
+---------------------------+
           |
           v
+---------------------------+
| Acoustic Model (声学模型)   |
| e.g., Tacotron, FastSpeech |
+---------------------------+
           |
           v
+---------------------------+
| Vocoder (声码器)            |
| e.g., HiFi-GAN, WaveGlow   |
+---------------------------+
           |
           v
+---------------------------+
| Audio Output (音频输出)    |
+---------------------------+


```


# 参考 
[pytorch audio](https://github.com/pytorch/audio/tree/main/examples)

[speechbrain](https://speechbrain.github.io/)