Run the following commands to set up the environment. If these do not work, you can try another option from the social jax github: https://github.com/cooperativex/SocialJax/blob/main/README.md 

First, cd into the SocialJax directory:

```bash
cd SocialJax
```

All the following commands should be run from the SocialJax directory. 

1. Conda
   ```bash
   conda create -n SocialJax python=3.10
   conda activate SocialJax
   ```

2. Install requirements
     ```bash
     pip install -r requirements.txt
     pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     ```
     ```bash
     export PYTHONPATH=./socialjax:$PYTHONPATH
     ```

Then, to run our code:

3. Run code
     ```bash
     ippo_harvest_better_comm.py
     ```

The seeds for the training runs and other hyperparameters are set at the beginning of the file.    

4. Then to generate the remaining plots
        ```bash
        python plots.py
        ```