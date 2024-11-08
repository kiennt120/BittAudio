module.exports = {
    apps : [{
      name: 'miner_uid',
      script: 'neurons/miner.py',
      interpreter: '/root/miniconda3/envs/sn50/bin/python3',
      args: [
        '--wallet.name', 'ckdevops4',
        '--wallet.hotkey', 'hkx',
        '--netuid', '50',
        '--axon.port', '700x',
        '--axon.external_port', '',
        '--music_model', 'facebook/musicgen-medium',
        '--logging.trace'
      ].join(' ')
    }]
}