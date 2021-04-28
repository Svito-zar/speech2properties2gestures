def hparam_options(hparams, trial):

    hparams.decoder["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    hparams.decoder["n_layers"] = trial.suggest_categorical("n_layers", [1, 2, 3])
    hparams.decoder["dropout"] = trial.suggest_uniform("dropout", 0.1, 0.6)
    
    hparams.text_enc["hidden_dim"] = trial.suggest_categorical("txt_hidden_dim", [32, 64, 96, 128])
    hparams.text_enc["output_dim"] = trial.suggest_categorical("txt_output_dim", [16, 32, 64, 96])
    hparams.text_enc["n_layers"] = trial.suggest_categorical("txt_n_layers", [1, 2, 3])
    hparams.text_enc["kernel_size"] = trial.suggest_categorical("txt_kernel_size", [ 3, 5, 7])
    hparams.text_enc["dropout"] = trial.suggest_uniform("txt_dropout", 0.1, 0.6)
    
    hparams.audio_enc["hidden_dim"] = trial.suggest_categorical("audio_hidden_dim", [32, 64, 96, 128])
    hparams.audio_enc["output_dim"] = trial.suggest_categorical("aud_out_dim", [16, 32, 64, 96])
    hparams.audio_enc["n_layers"] = trial.suggest_categorical("aud_n_layers", [1, 2, 3])
    hparams.audio_enc["kernel_size"] = trial.suggest_categorical("aud_kernel_size", [ 3, 5, 7])
    hparams.audio_enc["dropout"] = trial.suggest_uniform("aud_dropout", 0.1, 0.6)
    
    hparams.Loss["beta"] = trial.suggest_uniform("beta", 0.75, 0.9)
    hparams.Loss["gamma"] = trial.suggest_int("gamma", 2, 12)
    hparams.Loss["alpha"] = trial.suggest_uniform("alpha", 0.60, 0.85)

    hparams.lr = trial.suggest_loguniform("lr", 5e-5, 5e-3)
    hparams.batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 684])

    return hparams
