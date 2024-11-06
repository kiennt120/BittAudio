module.exports = {
    apps : [{
      name: 'miner_uid-...',
      script: 'neurons/miner.py',
      interpreter: '/root/miniconda3/envs/sn50/bin/python3',
      args: [
        '--wallet.name', '',
        '--wallet.hotkey', '',
        '--netuid', '50',
        '--axon.port', '',
        '--axon.external_port', '',
        '--music_model', 'facebook/musicgen-medium',
        '--logging.trace'
      ].join(' ')
    }]
}