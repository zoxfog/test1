#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
from os import getcwd
from os.path import basename, dirname
import pandas as pd
import numpy as np
import sys
import darts
from darts.models import NBEATSModel,BlockRNNModel,RNNModel,ExponentialSmoothing, TCNModel, TransformerModel,TFTModel



models = ['RNN','N-BEATS-G','N-BEATS-I','TCN']
def get_model(model_name, inlen, outlen, seed = 77):
    if model_name == 'RNN':

        model = BlockRNNModel(
            input_chunk_length=inlen,
            output_chunk_length = outlen,
            model="RNN",
            hidden_size=100,
            n_rnn_layers=2,
            dropout=0,
            batch_size=32,
            n_epochs=200,
            optimizer_kwargs={"lr": 0.001},
            model_name="Air_RNN",
            log_tensorboard=False,
            random_state= seed ,
            force_reset=True,
            save_checkpoints=False,
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
        
    elif model_name == "TCN":
        model = TCNModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            batch_size=32,
            n_epochs=200,
            model_name=model_name,
            optimizer_kwargs={"lr": 0.01},
            dropout=0,
            dilation_base=2,
            weight_norm=True,
            kernel_size=5,
            num_filters=3,            
            force_reset=True,
            random_state= seed ,
            save_checkpoints=False,
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
        

        
    elif model_name == 'N-BEATS-G':
        model = NBEATSModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            force_reset=True,
            n_epochs=200,
            nr_epochs_val_period=1,
            random_state= seed ,
            batch_size=32,
            model_name="nbeats_generic_run",
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },
        )

        
    elif model_name == 'N-BEATS-I':
        model = NBEATSModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            generic_architecture=False,
            num_blocks=3,
            num_layers=4,
            layer_widths=512,
            n_epochs=200,
            nr_epochs_val_period=1,
            force_reset=True,
            random_state= seed ,
            batch_size=32,
            model_name="nbeats_interpretable_run",
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },
        )
    elif model_name == 'TRANSFORMER':
        model = TransformerModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            batch_size=32,
            n_epochs=200,
            model_name="transformer_run",
            nr_epochs_val_period=1,
            optimizer_kwargs={"lr": 0.001},
            d_model=16,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0,
            activation="relu",
            random_state=seed,
            save_checkpoints=False,
            force_reset=True,
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
        
    elif model_name == 'TFT':
        model = TFTModel(
        input_chunk_length=inlen,
        output_chunk_length=outlen,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=300,
        add_relative_index=False,
        add_encoders=None,
        likelihood=None,
        random_state=seed,
        pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
                            
    
    else:
        print('TODO')
    return model


def get_modelM3(model_name, inlen, outlen, seed = 77):
    if model_name == 'RNN':

        model = BlockRNNModel(
            input_chunk_length=inlen,
            output_chunk_length = outlen,
            model="RNN",
            hidden_size=10,
            n_rnn_layers=2,
            dropout=0,
            batch_size=32,
            n_epochs=10,
            optimizer_kwargs={"lr": 0.001},
            model_name="Air_RNN",
            log_tensorboard=False,
            random_state= seed ,
            force_reset=True,
            save_checkpoints=False,
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
        
    elif model_name == "TCN":
        model = TCNModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            batch_size=32,
            n_epochs=10,
            model_name=model_name,
            optimizer_kwargs={"lr": 0.01},
            dropout=0,
            dilation_base=2,
            weight_norm=True,
            kernel_size=5,
            num_filters=3,            
            force_reset=True,
            random_state= seed ,
            save_checkpoints=False,
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
        

        
    elif model_name == 'N-BEATS-G':
        model = NBEATSModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            force_reset=True,
            n_epochs=100,
            nr_epochs_val_period=1,
            random_state= seed ,
            batch_size=32,
            model_name="nbeats_generic_run",
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },
        )

        
    elif model_name == 'N-BEATS-I':
        model = NBEATSModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            generic_architecture=False,
            num_blocks=3,
            num_layers=4,
            layer_widths=512,
            n_epochs=10,
            nr_epochs_val_period=1,
            force_reset=True,
            random_state= seed ,
            batch_size=32,
            model_name="nbeats_interpretable_run",
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },
        )
    elif model_name == 'TRANSFORMER':
        model = TransformerModel(
            input_chunk_length=inlen,
            output_chunk_length=outlen,
            batch_size=32,
            n_epochs=10,
            model_name="transformer_run",
            nr_epochs_val_period=1,
            optimizer_kwargs={"lr": 0.001},
            d_model=16,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0,
            activation="relu",
            random_state=seed,
            save_checkpoints=False,
            force_reset=True,
            pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
        
    elif model_name == 'TFT':
        model = TFTModel(
        input_chunk_length=inlen,
        output_chunk_length=outlen,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=10,
        add_relative_index=False,
        add_encoders=None,
        likelihood=None,
        random_state=seed,
        pl_trainer_kwargs={
              "accelerator": "gpu",
                "gpus": [0]
                },)
                            
    
    else:
        print('TODO')
    return model


