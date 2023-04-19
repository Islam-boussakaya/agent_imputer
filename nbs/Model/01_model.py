#!/usr/bin/env python
# coding: utf-8

# In[1]:


#| default_exp model.functions


# In[2]:


#| hide


# ## Agent Imputter Model
# > In this notebook we will load and train the model.

# In[3]:


#| export

import itertools
import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn import preprocessing

import torch
from torch.utils.data import DataLoader, Dataset

import wandb
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateFinder
from agent_imputter.model.agent_imputer import AgentImputerLightning


# ## Loading in the data

# ### Step 1: Load in pre-saved data 

# In[4]:


# |eval: false

# path to games files
games_path = Path("/home/user/Downloads/Data/games")
match_folder = tuple(
    name
    for name in os.listdir(games_path)
    if os.path.isdir(os.path.join(games_path, name))
)


# ### Step 2: Normalize the data

# In order to handle categorical features in our model, we decided to use label encoding instead of one hot encoding. The main reason is to reduce the dimensionality of the feature space. One hot encoding creates a binary vector for each category, leading to a high-dimensional sparse representation, which can be computationally expensive to handle. Label encoding, on the other hand, assigns a unique integer to each category, resulting in a lower-dimensional representation. By using label encoding, we can reduce the computational complexity of our model while still capturing the categorical information, in our case the number of features has been reduced from $N_{feature}$ = 60 to $N_{feature}$ = 5.

# In[5]:


#| export


def get_embedding_tensor_for_game(categories, idx):
    "compute embeddings of categorical features"

    def get_embedding(category_vec, idx):
        "compte embedding of one feature"
        num_classes = len(category_vec.unique())
        emb_size = math.floor(math.sqrt(num_classes))
        le = preprocessing.LabelEncoder()
        le.fit(category_vec.iloc[idx])
        cat = torch.tensor(
            np.array(le.transform(category_vec)).reshape(len(category_vec), 1)
        ).to(torch.int64)
        return cat

    cats = torch.tensor([])
    for cat in categories:
        new_cat = get_embedding(categories[cat], idx)
        cats = torch.cat((cats, new_cat), axis=1)
    return cats


# In[6]:


# emb_tensor = get_embedding_tensor_for_game(features_df[['position','event_type','team_on_ball','player_on_ball','goal_diff']],features_df.index)


# We employed the MinMaxScaler method to preprocess the numerical features in our dataset, which involves scaling the data to a specified range, between 0 and 1. 

# In[7]:


#| export


def preprocess_data(
    input_data: pd.DataFrame(),
) -> Tuple:
    """"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    time_scaler = MinMaxScaler(feature_range=(0, 1))

    emb_cats = get_embedding_tensor_for_game(
        input_data[
            ["position", "event_type", "team_on_ball", "player_on_ball", "goal_diff"]
        ],
        input_data.index,
    )
    scaler.fit(
        input_data[
            [
                "ballx",
                "prev_player_x",
                "next_player_x",
                "bally",
                "prev_player_y",
                "next_player_y",
                "av_player_x",
                "av_player_y",
            ]
        ]
    )
    time_scaler.fit(
        input_data[["time_since_last_pred", "prev_player_time", "next_player_time"]]
    )

    input_data_normalized = scaler.transform(
        input_data[
            [
                "ballx",
                "prev_player_x",
                "next_player_x",
                "bally",
                "prev_player_y",
                "next_player_y",
                "av_player_x",
                "av_player_y",
            ]
        ]
    )
    input_data_time = time_scaler.transform(
        input_data[["time_since_last_pred", "prev_player_time", "next_player_time"]]
    )

    input_data_normalized = np.concatenate(
        (input_data_normalized, input_data_time), axis=1
    )
    input_data_normalized = torch.cat(
        (torch.tensor(input_data_normalized), emb_cats), 1
    )

    label_data = torch.tensor(features_df[["label_x", "label_y"]].values)
    scaler.fit(label_data)
    label_data_normalized = scaler.transform(label_data)

    return input_data_normalized, label_data_normalized, scaler


# In[8]:


# input_data_normalized, label_data_normalized, scaler = preprocess_data(features_df)


# ### Step 3: Create seqeuences from the data

# In this section, our aim is to prepare the data in a format suitable for input into our model, which requires a tensor of shape ($N$ × $L$ × $I$), where $N$ is the number of agents, $L$ is the sequence length, and $I$ is the number of features. For our use case, we have 22 players ($N=22$), each with a sequence of 5 events ($L=5$), and 16 features per event ($I=16$), resulting in a tensor of shape ($22$ × $5$ × $16$). Additionally, we need to create a timestamps vector that we will pass to the model, which will also have shape ($22$ × $5$), corresponding to the 22 agents and 5 events.

# In[9]:


#| export


def split_sequences(
    sorted_whole_input_df: pd.DataFrame(),
    input_data_normalized: torch.tensor,
    label_data_normalized: torch.tensor,
    n_steps_in: int,
    n_steps_out: int,
):
    "Gets sequences of the previous and next x values for input data"

    time_scaler = RobustScaler()
    timestamps = torch.tensor(
        time_scaler.fit_transform(
            np.array(sorted_whole_input_df["event_time"]).reshape(-1, 1)
        )
    ).reshape(-1)

    # Define the number of previous and next tensors to include
    num_prev_tensors = n_steps_in
    num_next_tensors = n_steps_out

    # Create a list to store the resulting tensors
    X = []
    y = []
    ts = []
    for i in range(0, int(len(sorted_whole_input_df) / 22)):
        prev_indices = range(max(i - num_prev_tensors, 0), i)
        next_indices = range(
            i + 1, min(i + num_next_tensors + 1, int(len(sorted_whole_input_df) / 22))
        )
        idx = []

        if len(prev_indices) == 0:
            idx.extend([i, i, i])
        elif len(prev_indices) == 1:
            idx.extend([i - 1, i - 1, i])
        else:
            idx.extend([i - 2, i - 1, i])

        if len(next_indices) == 0:
            idx.extend([i, i])
        elif len(next_indices) == 1:
            idx.extend([i + 1, i + 1])
        else:
            idx.extend([i + 1, i + 2])

        l_x = []
        l_ts = []
        for j in idx:
            eve_df = sorted_whole_input_df[sorted_whole_input_df["event_num"] == j]
            l_x.append(input_data_normalized[eve_df.index[0] : eve_df.index[-1] + 1])
            l_ts.append(timestamps[eve_df.index[0] : eve_df.index[-1] + 1])

        # concatenate the tensors along a new dimension
        input_result_tensor = torch.stack(l_x, dim=1)
        ts_result_tensor = torch.stack(l_ts, dim=1)

        # convert to list of 22 tensors of shape (5, 16)
        input_tensor_list = [tensor.squeeze(0) for tensor in input_result_tensor]
        ts_tensor_list = [tensor.squeeze() for tensor in ts_result_tensor]
        ts_tensor_list = [
            torch.abs(ts_tensor - ts_tensor[2]) for ts_tensor in ts_tensor_list
        ]
        X.extend(input_tensor_list)
        ts.extend(ts_tensor_list)

    y = [tensor.squeeze() for tensor in label_data_normalized]
    return X, y, ts


# In[10]:


# X_ss, y_mm, ts = split_sequences(features_df,input_data_normalized, label_data_normalized, 2, 2)


# ## Generate and save model inputs for all games

# In[11]:


# |eval: false

whole_input = pd.DataFrame()
l_X_ss = []
l_y_mm = []
l_ts = []
for game in match_folder:
    print(game)
    features_df = pd.read_csv(Path(f"{games_path}/{game}/features.csv"))
    whole_input = pd.concat(
        [
            whole_input,
            features_df[
                ["match_id", "event_id", "player_id", "position", "label_x", "label_y"]
            ],
        ]
    )
    input_data_normalized, label_data_normalized, scaler = preprocess_data(features_df)
    X_ss, y_mm, ts = split_sequences(
        features_df, input_data_normalized, label_data_normalized, 2, 2
    )
    l_X_ss = [*l_X_ss, *X_ss]
    l_y_mm = [*l_y_mm, *y_mm]
    l_ts = [*l_ts, *ts]

l_event_id = whole_input["event_id"].values
l_match_id = whole_input["match_id"].values
l_player_id = whole_input["player_id"].values


# In[12]:


# save input df
# whole_input.to_csv("/home/user/Downloads/agent_imputter_output/17_pred/whole_input.csv")


# ## Split train and test Set

# In this section, we will split the dataset into two parts: the training set and the test set. To ensure consistency with the paper, we will use the same proportion of `91.2%` for the training set and `8.8%` for the test set. This division of the data will allow us to train the model on a sufficiently large sample while also ensuring that we have enough data left over to evaluate its performance on unseen data. By following this approach, we can ensure that our results are comparable to those reported in the paper.

# In[13]:


#| export


def get_train_test_split(
    X_ss, y_mm, ts, l_event_id, l_match_id, l_player_id, train_size=0.912, shuffle=False
):
    "Get train and test split"
    train_size = int((len(X_ss) / 22) * train_size) * 22
    (
        X_train,
        X_test,
        y_train,
        y_test,
        ts_train,
        ts_test,
        event_ids_train,
        event_ids_test,
        match_ids_train,
        match_ids_test,
        player_ids_train,
        player_ids_test,
    ) = train_test_split(
        X_ss,
        y_mm,
        ts,
        l_event_id,
        l_match_id,
        l_player_id,
        random_state=42,
        shuffle=shuffle,
        train_size=train_size,
    )

    events_ids = [event_ids_train, event_ids_test]
    match_ids = [match_ids_train, match_ids_test]
    player_ids = [player_ids_train, player_ids_test]
    return (
        tuple(X_train),
        tuple(X_test),
        torch.tensor(np.array(y_train)),
        torch.tensor(np.array(y_test)),
        tuple(ts_train),
        tuple(ts_test),
        events_ids,
        match_ids,
        player_ids,
    )


# In[14]:


(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_ts,
    X_test_ts,
    events_ids,
    match_ids,
    player_ids,
) = get_train_test_split(l_X_ss, l_y_mm, l_ts, l_event_id, l_match_id, l_player_id)


# ### Convert to pytorch Dataloaders

# In this section we will convert our dataset to pytorch dataset and then to pytorch dataloader. 

# In[15]:


#| export


class series_data(Dataset):
    "Convert into chunks of 22 players and sequences of 5 with all features"

    def __init__(self, x, y, t, feature_num):
        self.x = torch.stack(x).reshape(int(len(x) / 22), 22, 5, feature_num)
        self.y = y.clone().detach().float().reshape(int(len(x) / 22), 22, 2)
        self.t = torch.stack(t).reshape(int(len(x) / 22), 22, 5)
        self.len = int(len(x) / 22)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx]

    def __len__(self):
        return self.len


# In[16]:


# same batch size used in paper.
BATCH_SIZE = 128


# In[17]:


# Put into format of 22 event sequences for an event (representing each player) and put into data loaders
train_data = series_data(X_train, y_train, X_train_ts, 16)
test_data = series_data(X_test, y_test, X_test_ts, 16)

train_loader = DataLoader(
    train_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=12
)
test_loader = DataLoader(
    test_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=12
)


# ## Load and Run Model

# In[22]:


# |eval: false

wandb_logger = WandbLogger(project="agent_imputter")
wandb_logger.experiment.config["batch_size"] = BATCH_SIZE


# In[23]:


# initiate model callbacks
checkpoint_callback = ModelCheckpoint(monitor="validation loss", mode="min")


# In[24]:


# initiate model
model = AgentImputerLightning()


# In[25]:


# initiate trainer
trainer = pl.Trainer(
    callbacks=[checkpoint_callback, LearningRateFinder(0.01, 0.001)],
    max_epochs=400,
    min_epochs=1,
    accelerator="auto",
    devices="auto",
    logger=wandb_logger,
)


# In[26]:


# |eval: false

trainer.fit(model, train_loader, test_loader)


# In[28]:


# |eval: false

wandb.finish()


# ## predict and save train and test prediction 

# In[ ]:


# |eval: false

train_loader = DataLoader(train_data, shuffle=False, batch_size=len(train_data) * 22)
test_loader = DataLoader(test_data, shuffle=False, batch_size=len(test_data) * 22)

_train_p = trainer.predict(model, train_loader)
_test_p = trainer.predict(model, test_loader)


# ### Save data

# In[ ]:


# |eval: false

train_df = pd.DataFrame()
train_df["match_id"], train_df["event_id"], train_df["player_id"] = (
    match_ids[0],
    events_ids[0],
    player_ids[0],
)

test_df = pd.DataFrame()
test_df["match_id"], test_df["event_id"], test_df["player_id"] = (
    match_ids[1],
    events_ids[1],
    player_ids[1],
)

train_p = scaler.inverse_transform(_train_p[0].reshape(len(train_data) * 22, 2))
test_p = scaler.inverse_transform(_test_p[0].reshape(len(test_data) * 22, 2))
train_y = scaler.inverse_transform(y_train)
test_y = scaler.inverse_transform(y_test)

train_df["pred_x"], train_df["pred_y"] = train_p[:, 0], train_p[:, 1]
train_df["act_x"], train_df["act_y"] = train_y[:, 0], train_y[:, 1]

test_df["pred_x"], test_df["pred_y"] = test_p[:, 0], test_p[:, 1]
test_df["act_x"], test_df["act_y"] = test_y[:, 0], test_y[:, 1]


# In[ ]:


# |eval: false

train_df.to_csv("/home/user/Downloads/agent_imputter_output/17_pred/train_df_1.csv")
test_df.to_csv("/home/user/Downloads/agent_imputter_output/17_pred/test_df_1.csv")


# In[ ]:





# In[ ]:


#| hide
from nbdev import nbdev_export

nbdev_export()


# In[ ]:




