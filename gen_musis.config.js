module.exports = {
    apps : [{
      name: 'gen_music',
      script: 'serve.py',
      interpreter: '/root/miniconda3/envs/sn50/bin/python3',
      args: [
        "--music_path", 'facebook/musicgen-large',
        "--sample_rate", "44100",
        "--output_path", "outputs",
        "--port", "5000",
      ].join(' ')
    }]
}