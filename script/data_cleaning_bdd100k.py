{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.13",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "sourceId": 442057,
          "sourceType": "datasetVersion",
          "datasetId": 161598
        }
      ],
      "dockerImageVersionId": 30646,
      "isInternetEnabled": true,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": false
    },
    "colab": {
      "name": "data_cleaning_bdd100k",
      "provenance": []
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "source": [
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "import kagglehub\n",
        "solesensei_solesensei_bdd100k_path = kagglehub.dataset_download('solesensei/solesensei_bdd100k')\n",
        "\n",
        "print('Data source import complete.')\n"
      ],
      "metadata": {
        "id": "ITUfHdKyIClf"
      },
      "cell_type": "code",
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "import shutil\n",
        "\n",
        "# Load JSON file into a DataFrame\n",
        "JSON_PATH_TRAIN = '/kaggle/input/solesensei_bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'\n",
        "JSON_PATH_VAL = '/kaggle/input/solesensei_bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'\n",
        "df_train = pd.read_json(JSON_PATH_TRAIN)\n",
        "\n",
        "df_val = pd.read_json(JSON_PATH_VAL)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T15:34:09.319206Z",
          "iopub.execute_input": "2024-02-17T15:34:09.319615Z",
          "iopub.status.idle": "2024-02-17T15:34:49.748744Z",
          "shell.execute_reply.started": "2024-02-17T15:34:09.319584Z",
          "shell.execute_reply": "2024-02-17T15:34:49.74757Z"
        },
        "trusted": true,
        "id": "dZffdBkiIClh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_train['timeofday_values'] = df_train['attributes'].apply(lambda x: x.get('timeofday', None))\n",
        "df_val['timeofday_values'] = df_val['attributes'].apply(lambda x: x.get('timeofday', None))\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T15:34:49.751005Z",
          "iopub.execute_input": "2024-02-17T15:34:49.751383Z",
          "iopub.status.idle": "2024-02-17T15:34:49.833601Z",
          "shell.execute_reply.started": "2024-02-17T15:34:49.751352Z",
          "shell.execute_reply": "2024-02-17T15:34:49.832459Z"
        },
        "trusted": true,
        "id": "8E65dMi4ICli"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_train.head(2)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T15:34:49.835174Z",
          "iopub.execute_input": "2024-02-17T15:34:49.835554Z",
          "iopub.status.idle": "2024-02-17T15:34:49.890485Z",
          "shell.execute_reply.started": "2024-02-17T15:34:49.835525Z",
          "shell.execute_reply": "2024-02-17T15:34:49.889335Z"
        },
        "trusted": true,
        "id": "iYWwL8cvICli"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_val.head(2)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T15:34:49.892907Z",
          "iopub.execute_input": "2024-02-17T15:34:49.893257Z",
          "iopub.status.idle": "2024-02-17T15:34:49.993755Z",
          "shell.execute_reply.started": "2024-02-17T15:34:49.893228Z",
          "shell.execute_reply": "2024-02-17T15:34:49.992597Z"
        },
        "trusted": true,
        "id": "NjMNQxx9ICli"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_train['weather'] = df_train['attributes'].apply(lambda x: x.get('weather', None))\n",
        "df_val['weather'] = df_val['attributes'].apply(lambda x: x.get('weather', None))\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:15:12.861469Z",
          "iopub.execute_input": "2024-02-17T16:15:12.861863Z",
          "iopub.status.idle": "2024-02-17T16:15:13.201122Z",
          "shell.execute_reply.started": "2024-02-17T16:15:12.861831Z",
          "shell.execute_reply": "2024-02-17T16:15:13.199686Z"
        },
        "trusted": true,
        "id": "-vpURC58ICli"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_train.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:15:16.559014Z",
          "iopub.execute_input": "2024-02-17T16:15:16.559438Z",
          "iopub.status.idle": "2024-02-17T16:15:16.671033Z",
          "shell.execute_reply.started": "2024-02-17T16:15:16.559403Z",
          "shell.execute_reply": "2024-02-17T16:15:16.669801Z"
        },
        "trusted": true,
        "id": "7OKk0OvSIClj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_val.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:15:22.534141Z",
          "iopub.execute_input": "2024-02-17T16:15:22.534549Z",
          "iopub.status.idle": "2024-02-17T16:15:22.768814Z",
          "shell.execute_reply.started": "2024-02-17T16:15:22.53452Z",
          "shell.execute_reply": "2024-02-17T16:15:22.767576Z"
        },
        "trusted": true,
        "id": "FkBW7mtZIClj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(df_train['timeofday_values'].value_counts())\n",
        "print(df_val['timeofday_values'].value_counts())\n",
        "\n",
        "print(df_train['weather'].value_counts())\n",
        "print(df_val['weather'].value_counts())"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:15:48.784439Z",
          "iopub.execute_input": "2024-02-17T16:15:48.784875Z",
          "iopub.status.idle": "2024-02-17T16:15:48.873092Z",
          "shell.execute_reply.started": "2024-02-17T16:15:48.784843Z",
          "shell.execute_reply": "2024-02-17T16:15:48.87176Z"
        },
        "trusted": true,
        "id": "ZZcSaEnLIClj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ROOT = os.getcwd()\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:19:12.528144Z",
          "iopub.execute_input": "2024-02-17T16:19:12.528818Z",
          "iopub.status.idle": "2024-02-17T16:19:12.535218Z",
          "shell.execute_reply.started": "2024-02-17T16:19:12.528773Z",
          "shell.execute_reply": "2024-02-17T16:19:12.533791Z"
        },
        "trusted": true,
        "id": "e17blmrXIClj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_val['timeofday'] =df_val['timeofday_values']\n",
        "val_labels= df_val.drop([\"labels\", \"timeofday_values\" ,\"timestamp\", 'attributes'], axis=1, errors='ignore')"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:19:13.547233Z",
          "iopub.execute_input": "2024-02-17T16:19:13.548031Z",
          "iopub.status.idle": "2024-02-17T16:19:13.56985Z",
          "shell.execute_reply.started": "2024-02-17T16:19:13.547993Z",
          "shell.execute_reply": "2024-02-17T16:19:13.568404Z"
        },
        "trusted": true,
        "id": "ONPsYYqNIClj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "val_labels.to_csv(ROOT+\"/val_labels.csv\", index=False)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:19:14.846165Z",
          "iopub.execute_input": "2024-02-17T16:19:14.846584Z",
          "iopub.status.idle": "2024-02-17T16:19:14.899742Z",
          "shell.execute_reply.started": "2024-02-17T16:19:14.846553Z",
          "shell.execute_reply": "2024-02-17T16:19:14.898266Z"
        },
        "trusted": true,
        "id": "VgDyTj1-IClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def list_files_in_dataframe(root_folder):\n",
        "    file_data = {'folder': [], 'subfolder': [], 'filename': []}\n",
        "\n",
        "    for folder in os.listdir(root_folder):\n",
        "        folder_path = os.path.join(root_folder, folder)\n",
        "        if os.path.isdir(folder_path):\n",
        "            for subfolder in os.listdir(folder_path):\n",
        "                subfolder_path = os.path.join(folder_path, subfolder)\n",
        "                if os.path.isdir(subfolder_path):\n",
        "                    for filename in os.listdir(subfolder_path):\n",
        "                        file_data['folder'].append(folder)\n",
        "                        file_data['subfolder'].append(subfolder)\n",
        "                        file_data['filename'].append(filename)\n",
        "\n",
        "    df = pd.DataFrame(file_data)\n",
        "    return df\n",
        "\n",
        "root_folder = \"/kaggle/input/solesensei_bdd100k/bdd100k/bdd100k/images/100k\"\n",
        "train_test_df = list_files_in_dataframe(root_folder)\n",
        "print(train_test_df.head())\n",
        "train_test_df.to_csv(ROOT + '/train_test_filenames.csv')"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:19:16.831621Z",
          "iopub.execute_input": "2024-02-17T16:19:16.832Z",
          "iopub.status.idle": "2024-02-17T16:19:40.560323Z",
          "shell.execute_reply.started": "2024-02-17T16:19:16.831969Z",
          "shell.execute_reply": "2024-02-17T16:19:40.558486Z"
        },
        "trusted": true,
        "id": "GVYqb3pMIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def merge_dataframe(df_additional, df_base):\n",
        "    \"\"\"\n",
        "    Merges and adds 'timeofday_values' from df_additional into df_base based on matching filenames.\n",
        "\n",
        "    Parameters:\n",
        "    - df_additional (pd.DataFrame): The DataFrame containing additional information.\n",
        "    - df_base (pd.DataFrame): The base DataFrame.\n",
        "\n",
        "    Returns:\n",
        "    - pd.DataFrame: Merged DataFrame with 'timeofday' column added to df_base.\n",
        "    \"\"\"\n",
        "\n",
        "    # Merge DataFrames based on 'name' and 'filename'\n",
        "    merged_df = pd.merge(df_base, df_additional, left_on='filename', right_on='name', how='left')\n",
        "\n",
        "    # Fill missing values in 'timeofday' with values from 'timeofday_values'\n",
        "    merged_df['timeofday'] = merged_df['timeofday_values']\n",
        "\n",
        "    # Drop the redundant 'filename' and 'timeofday_values' columns\n",
        "    merged_df = merged_df.drop(['filename', 'timeofday_values', \"labels\", \"timestamp\", 'attributes'], axis=1, errors='ignore')\n",
        "\n",
        "    return merged_df\n",
        "\n",
        "\n",
        "df_merged = merge_dataframe(df_train, train_test_df)\n",
        "df_merged\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:20:23.603849Z",
          "iopub.execute_input": "2024-02-17T16:20:23.604313Z",
          "iopub.status.idle": "2024-02-17T16:20:24.108629Z",
          "shell.execute_reply.started": "2024-02-17T16:20:23.604274Z",
          "shell.execute_reply": "2024-02-17T16:20:24.107091Z"
        },
        "trusted": true,
        "id": "KbiJmWqjIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_merged.info()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:20:33.5973Z",
          "iopub.execute_input": "2024-02-17T16:20:33.597687Z",
          "iopub.status.idle": "2024-02-17T16:20:33.700862Z",
          "shell.execute_reply.started": "2024-02-17T16:20:33.597658Z",
          "shell.execute_reply": "2024-02-17T16:20:33.699265Z"
        },
        "trusted": true,
        "id": "Fo1ed9CMIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_merged.to_csv(ROOT +'/labels.csv', index=False)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:20:43.842674Z",
          "iopub.execute_input": "2024-02-17T16:20:43.843046Z",
          "iopub.status.idle": "2024-02-17T16:20:44.305167Z",
          "shell.execute_reply.started": "2024-02-17T16:20:43.843018Z",
          "shell.execute_reply": "2024-02-17T16:20:44.303926Z"
        },
        "trusted": true,
        "id": "RUEbYIvcIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test= pd.read_csv(\"/kaggle/working/labels.csv\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:21:35.117361Z",
          "iopub.execute_input": "2024-02-17T16:21:35.117792Z",
          "iopub.status.idle": "2024-02-17T16:21:35.247969Z",
          "shell.execute_reply.started": "2024-02-17T16:21:35.117759Z",
          "shell.execute_reply": "2024-02-17T16:21:35.246614Z"
        },
        "trusted": true,
        "id": "V1t4MhMeIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test.head()\n",
        "test.dropna(inplace=True)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:21:39.368557Z",
          "iopub.execute_input": "2024-02-17T16:21:39.368942Z",
          "iopub.status.idle": "2024-02-17T16:21:39.427815Z",
          "shell.execute_reply.started": "2024-02-17T16:21:39.368913Z",
          "shell.execute_reply": "2024-02-17T16:21:39.426303Z"
        },
        "trusted": true,
        "id": "7NR9X0-lIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test.folder.value_counts()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:21:46.285744Z",
          "iopub.execute_input": "2024-02-17T16:21:46.286567Z",
          "iopub.status.idle": "2024-02-17T16:21:46.306263Z",
          "shell.execute_reply.started": "2024-02-17T16:21:46.286532Z",
          "shell.execute_reply": "2024-02-17T16:21:46.30476Z"
        },
        "trusted": true,
        "id": "9Qdp65uxIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test.to_csv(ROOT +\"/training_labels.csv\", index=False)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:21:51.059667Z",
          "iopub.execute_input": "2024-02-17T16:21:51.060081Z",
          "iopub.status.idle": "2024-02-17T16:21:51.344504Z",
          "shell.execute_reply.started": "2024-02-17T16:21:51.060049Z",
          "shell.execute_reply": "2024-02-17T16:21:51.343064Z"
        },
        "trusted": true,
        "id": "W0RaYrr_IClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!rm -rf /kaggle/working/labels.csv"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:21:57.580819Z",
          "iopub.execute_input": "2024-02-17T16:21:57.58129Z",
          "iopub.status.idle": "2024-02-17T16:21:59.078343Z",
          "shell.execute_reply.started": "2024-02-17T16:21:57.581252Z",
          "shell.execute_reply": "2024-02-17T16:21:59.07675Z"
        },
        "trusted": true,
        "id": "LOWPoe1zIClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def reorganize_images_based_on_timeofday(csv_path, source_folder, destination_folder):\n",
        "    \"\"\"\n",
        "    Organizes images into ()'testA' and 'testB' subfolders based on timeofday information.\n",
        "\n",
        "    Parameters:\n",
        "    - csv: containing columns 'name' and 'timeofday'.\n",
        "    - source_folder (str): Path to the source folder containing the images.\n",
        "    - destination_folder (str): Path to the destination folder where 'valA' and 'valB' subfolders will be created.\n",
        "\n",
        "    Returns:\n",
        "    - None\n",
        "    \"\"\"\n",
        "    df = pd.read_csv(csv_path)\n",
        "\n",
        "    for index, row in df.iterrows():\n",
        "        filename = row['name']\n",
        "        is_day = row['timeofday'] == \"daytime\"\n",
        "        is_night = row['timeofday'] == \"night\"\n",
        "        is_clear = row['weather'] == 'clear'\n",
        "\n",
        "        if is_day and is_clear:\n",
        "            timeofday_folder = 'testA' if row['subfolder'].startswith('test') else 'trainA'\n",
        "        elif is_night and is_clear:\n",
        "            timeofday_folder = 'testB' if row['subfolder'].startswith('test') else 'trainB'\n",
        "        else:\n",
        "            continue\n",
        "\n",
        "        source_path = os.path.join(source_folder, row['folder'], row['subfolder'], filename)\n",
        "        destination_path = os.path.join(destination_folder, timeofday_folder, filename)\n",
        "\n",
        "        # Create destination folder if it doesn't exist\n",
        "        os.makedirs(os.path.dirname(destination_path), exist_ok=True)\n",
        "\n",
        "        # Move the file to the new location\n",
        "        shutil.copy(source_path, destination_path)\n",
        "\n",
        "\n",
        "\n",
        "csv_path = \"/kaggle/working/training_labels.csv\"\n",
        "source_folder =  \"/kaggle/input/solesensei_bdd100k/bdd100k/bdd100k/images/100k\"\n",
        "destination_folder = \"/kaggle/working/night_to_day\"\n",
        "reorganize_images_based_on_timeofday(csv_path, source_folder, destination_folder)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:23:29.486035Z",
          "iopub.execute_input": "2024-02-17T16:23:29.486496Z",
          "iopub.status.idle": "2024-02-17T16:26:45.054498Z",
          "shell.execute_reply.started": "2024-02-17T16:23:29.486462Z",
          "shell.execute_reply": "2024-02-17T16:26:45.053254Z"
        },
        "trusted": true,
        "id": "fszHTXL5IClk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "val =  \"/kaggle/input/solesensei_bdd100k/bdd100k/bdd100k/images/100k/val\""
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:26:45.056556Z",
          "iopub.execute_input": "2024-02-17T16:26:45.056908Z",
          "iopub.status.idle": "2024-02-17T16:26:45.062519Z",
          "shell.execute_reply.started": "2024-02-17T16:26:45.056878Z",
          "shell.execute_reply": "2024-02-17T16:26:45.061411Z"
        },
        "trusted": true,
        "id": "lD-yAxquICll"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def organize_images(csv, source_folder, destination_folder):\n",
        "    \"\"\"\n",
        "    Organizes images into 'valA' and 'valB' subfolders based on timeofday information.\n",
        "\n",
        "    Parameters:\n",
        "    - csv: containing columns 'name' and 'timeofday'.\n",
        "    - source_folder (str): Path to the source folder containing the images.\n",
        "    - destination_folder (str): Path to the destination folder where 'valA' and 'valB' subfolders will be created.\n",
        "\n",
        "    Returns:\n",
        "    - None\n",
        "    \"\"\"\n",
        "    df = pd.read_csv(csv)\n",
        "    # Create destination subfolders if they don't exist\n",
        "    valA_folder = os.path.join(destination_folder, 'valA')\n",
        "    valB_folder = os.path.join(destination_folder, 'valB')\n",
        "\n",
        "    os.makedirs(valA_folder, exist_ok=True)\n",
        "    os.makedirs(valB_folder, exist_ok=True)\n",
        "\n",
        "    for index, row in df.iterrows():\n",
        "        image_name = row['name']\n",
        "        timeofday = row['timeofday']\n",
        "        is_clear = row['weather'] == 'clear'\n",
        "        if timeofday == 'daytime':\n",
        "            destination_path = os.path.join(valA_folder, image_name)\n",
        "        elif timeofday == 'night':\n",
        "            destination_path = os.path.join(valB_folder, image_name)\n",
        "        else:\n",
        "            continue\n",
        "\n",
        "        source_path = os.path.join(source_folder, image_name)\n",
        "\n",
        "        # Move the file to the appropriate subfolder\n",
        "        shutil.copy(source_path, destination_path)\n",
        "\n",
        "\n",
        "val_csv = \"/kaggle/working/val_labels.csv\"\n",
        "organize_images(val_csv, val, destination_folder)\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-17T16:26:45.064565Z",
          "iopub.execute_input": "2024-02-17T16:26:45.06499Z",
          "iopub.status.idle": "2024-02-17T16:27:31.9206Z",
          "shell.execute_reply.started": "2024-02-17T16:26:45.064956Z",
          "shell.execute_reply": "2024-02-17T16:27:31.918282Z"
        },
        "trusted": true,
        "id": "0VXza-8rICll"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Compute bias ratio for 'timeofday'\n",
        "timeofday_bias_ratio = df_train['timeofday'].value_counts(normalize=True)\n",
        "\n",
        "# Compute bias ratio for 'weather'\n",
        "# weather_bias_ratio = df['weather'].value_counts(normalize=True)\n",
        "\n",
        "# Print the bias ratios\n",
        "print(\"Bias Ratio for Time of Day:\")\n",
        "print(timeofday_bias_ratio)\n",
        "# print(\"\\nBias Ratio for Weather:\")\n",
        "# print(weather_bias_ratio)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2024-02-03T15:50:42.100184Z",
          "iopub.status.idle": "2024-02-03T15:50:42.10099Z",
          "shell.execute_reply.started": "2024-02-03T15:50:42.100634Z",
          "shell.execute_reply": "2024-02-03T15:50:42.100663Z"
        },
        "trusted": true,
        "id": "Gt0TLl-FICll"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "euzBg8v8ICll"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}