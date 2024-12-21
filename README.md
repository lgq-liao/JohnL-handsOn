# JohnL-handsOn

My NLP hand's on
**Note: To update xxx with your actual path for the following commands**



## Task 1 

1. **clone the repository into your testing PC**

```bash
    $ git clone https://github.com/lgq-liao/JohnL-handsOn.git
```

2. **installation**
```bash
    $ cd xxx/JohnL-handsOn/
    $ ./dependance_install.sh

```

# Task 2b

1. **launch the API**

```bash
    $ cd xxx/JohnL-handsOn/asr
    $ python asr_api.py
```

2. **test the API with your browser**

Launch your browser and enter the following link into your brower following by hit the **Enter** key : 

`http://localhost:8001/ping`

or click the link [ping](http://localhost:8001/ping)



# Task 2c

1. **launch the API**

```bash
    $ cd xxx/JohnL-handsOn/asr
    $ python asr_api.py
```

2. **test the API with my script**

```bash
    $ cd xxx/JohnL-handsOn/asr
    $ python python cv-decode.py -a audio_sample/taken_clip.wav
```

Alternatively, you can test it to use your own method,eg by curl

# Task 2d

1. **copy the cv-valid-dev audios and csv file from common voice folder to JohnL-handsOn/asr/**

```bash
    $ cd xxx/JohnL-handsOn/asr

    $ cp -r xxx/cv-valid-dev ./cv-valid-dev/ 

```

2. **execute cv-decode.py to call API to update the cv-valid-dev.csv file**

```bash
    $ cd xxx/JohnL-handsOn/asr
    $ python cv-decode.py -af ./cv-valid-dev
```

# Task 2e

1. **Build the Docker image**

```bash
    $ cd xxx/JohnL-handsOn/asr
    $ docker build -t asr-api .
```

2. **Running the Docker container with GPU support**
   
   **NOTE**: For GPU access, need root privileges if not config docker in rootless mode as following command. 

   For more details about how to enable GPU support (CUDA) in your Docker container, pleaase refer to [How to run Docker](asr/How%20to%20run%20Docker.md)


```bash

   $ sudo docker run --gpus all --rm -p 8001:8001 asr-api # run with GPU
   $ docker run --rm -p 8001:8001 asr-api                 # run with CPU
```