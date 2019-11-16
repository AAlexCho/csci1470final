
#!/usr/bin/env python

import logging
import numpy
import sys
import os
import importlib


try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print ("No plotting extension available.")


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build datastream
    path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/training")
    valid_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/validation")
    vocab_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt")

    ds, train_stream = data.setup_datastream(path, vocab_path, config)
    _, valid_stream = data.setup_datastream(valid_path, vocab_path, config)

    dump_path = os.path.join("model_params", model_name+".pkl")

    # Build model
    m = config.Model(config, ds.vocab_size)

    # Build the Blocks stuff for training
    model = Model(m.sgd_cost)

    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=model.parameters)

    extensions = [
            TrainingDataMonitoring(
                [v for l in m.monitor_vars for v in l],
                prefix='train',
                every_n_batches=config.print_freq)
    ]
    if config.save_freq is not None and dump_path is not None:
        extensions += [
            SaveLoadParams(path=dump_path,
                           model=model,
                           before_training=True,
                           after_training=True,
                           after_epoch=True,
                           every_n_batches=config.save_freq)
        ]
    if valid_stream is not None and config.valid_freq != -1:
        extensions += [
            DataStreamMonitoring(
                [v for l in m.monitor_vars_valid for v in l],
                valid_stream,
                prefix='valid',
                every_n_batches=config.valid_freq),
        ]
    if plot_avail:
        plot_channels = [['train_' + v.name for v in lt] + ['valid_' + v.name for v in lv]
                         for lt, lv in zip(m.monitor_vars, m.monitor_vars_valid)]
        extensions += [
            Plot(document='deepmind_qa_'+model_name,
                 channels=plot_channels,
                 # server_url='http://localhost:5006/', # If you need, change this
                 every_n_batches=config.print_freq)
        ]
    extensions += [
            Printing(every_n_batches=config.print_freq,
                     after_epoch=True),
            ProgressBar()
    ]

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    # Run the model !
    main_loop.run()
    main_loop.profile.report()

