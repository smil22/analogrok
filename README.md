# ANALOGROK

## Summary

This project presents a simple use case where a [short language model](https://en.wikipedia.org/wiki/Small_language_model) (SLM)
based on the [transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) architecture exhibits the unfamous
phenomenon of [grokking](https://en.wikipedia.org/wiki/Grokking_(machine_learning) through a basic [analogical reasoning mechanism](https://medium.com/@dickson.lukose/analogical-reasoning-d432b7105725).


## Principle

A rule has been set. Each time the model sees a particular letter, the prior one must be repeated to complete the input.
We defined basically three set of characters: the **lowercases ({'a', ... , 'z'})**, the **uppercases ({'A', ... , 'Z'})** and the 
**repetition characters, that is the particular letters ({'!', '?'})**. For exemple: **'a!' -> 'a'** or **'A?' -> 'A'**.

- **Training phase**: lowercases are only associated with '!' (**{'a', ... , 'z'} + '!'**) and 
uppercases only with '?' (**{'A', ..., 'Z'} + '?'**)
- **Test phase**: the associations are inverted, basically **{'a', ... , 'z'} + '?'** and **{'A', ... , 'Z'} + '!'**


## Goal

By putting the repetition characters in **analogous contexts**, we intended to check the model ability to 
**semantically drew them closer to each other** in a way that goes beyond the simple memorization (grokking) for an ultimate goal of 
seeing him associate whatever repetition character to whatever set of characters either lowercases or uppercases group.


## Observations and prospects

The model seems able to do it as you can see on the t-SNE projection of the embeddings space (check the **output** folders) ! 
That throws importants questions regarding not only the conditions of grokking but also the type of data and model size needed
to perform it. 
More investigations are to come. Until there, feel free to try it or test it by yourself and make suggestions or corrections.


## Files

 - **train.py** : run it if you want to re-train the model on the data.
 - **tools.py** : it contains the functions and methods used for data generation, model evaluation, ...
 - **components.py**: you will find there the components of the transformer based achitecture.
 - **test.py**: the model is saved after training. Run this file if you just want to load it and check its answers.

 
## Versions
 
 We ran three different versions of the use case, each one with its particularity. Check the README file of each version for more details.
 The versions are totally independent from each other.


## Installation

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/smil22/analogrok.git
cd ANALOGROK
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the environment

- On Linux/macOS

```bash
source venv/bin/activate
```
- On Windows (PowerShell)

```bash
.\venv\Scripts\Activate.ps1
```
- On Windows (cmd.exe)

```bash
.\venv\Scripts\activate.bat
```

### 4. Install dependencies

Run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the project

Navigate to the version folder you want and run the file you prefer, for example:

```bash
cd analogrok_2g[samples]
python train.py
```

### 6. To deactivate the created environment (optional)

```bash
deactivate
```
