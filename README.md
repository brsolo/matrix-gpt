# MatrixGPT

MatrixGPT is a 10.8 million parameter GPT built from scratch. It trains on the three Matrix movie transcripts ([tinymatrix.txt](text/tinymatrix.txt)) and is able to write its own script (for example [matrix_out](text\matrix_out.txt)).

The implementation is fully contained in [model.py](model.py) (~300 lines of code) which includes training of the transformer and generation of new text. The model trains 10.8 million parameters. On a NVIDIA GeForce RTX 3090, the training and generation took about 24 minutes.

The architecture of the model is diagrammed below:

- batched samples are tokenized using a simple character-based model
- the tokens are embedded in a 384 dimension matrix
- the position of each token in the sample is encoded in a separate 384 dimension matrix and added
- the output is fed into a block several times where each block contains:
    - a multi-head attention layer including a linear, dropout, and layer norm layer
    - the output of the multi-head attention layer is added back to its input
    - it then flows through a feedforward layer consisting of linear, ReLU, linear, dropout and layer norm layers
    - its input and then added back to th e output
- after passing several times through the block, the output is passed through layer norm and linear layers resulting in logits
- applying softmax to the logits produces probabilities that can be used for generation
- in training, the cross-entropy loss is computed from the logits and known targets (not shown in diagram), and AdamW is used to optimize the parameters

<br>

![diagram](assets/diagram.svg)


## Installation

If you want to play around with MatrixGPT, simply use:

```
git clone https://github.com/brsolo/matrix-gpt.git
cd matrix-gpt
```

## Usage

To run the code, simply run it from the command line:

```
!python model.py
```

In the header of [model.py](model.py), there are several hyperparameters (default values shown) that can easily be changed:
- `batch_size = 64`
- `block_size = 256`
- `max_iters = 5000` sets the number of iterations that the model trains for
- `max_new_tokens = 5000` sets the number of tokens that are generated
- `iters_print_size = max_iters/100` sets how frequently updates are printed as the model optimizes
- `learning_rate = 3e-4` is the learning rate for the AdamW optimization
- `torch.manual_seed(1337)` is added for reproducibility
- `eval_iters = 200` is how many iterations are averaged in calculating the loss
- `n_embd = 384` is the number of embedding dimensions
- `num_heads = 6` is the numbers of self-attention layers in a single multi-head attention layer
- `n_layer = 6` is the number of blocks containing multi-head attention and feedforward layers
- `dropout = 0.2` sets the probability that an element will be zeroed in all dropout layers


## Selected lines generated by MatrixGPT

Neo claims to be Trinity:

>Cypher: You have nour heard. It's all right, Trin. But you're gonna hour, I think you're going to be. You know that matters is going anywhere.
>
>Cypher: Do you know what he was do.
>
>Trinity: No?
>
>Neo: My name is Trinity.
>
>Neo: Trinity. The Trinity? Trinity? The one that cracked the IRS d-base?

<br>

A motivating albiet confusing monologue from Roland:

>Roland: Damn it, we are here not on down run to be old. Did you get Neo out here, know that follow to get through the Frenchman would. Every story, Neo. It's the conection was supposed to bring to bad the Matrix, to honly prophecy was a built to slies him too fught through. I was in look for that world be in your here. It is.

<br>

Interesting discussion between Neo and Tank:

>Neo: Do you know what it is?
>
>Tank: The Matrix?
>
>Neo: The Trainman. That's whe trace the one that power to deny the Trainman, but the would you like the hard?. I think the Matrix, I can't figure what I believer here to way out.

<br>

Neo and Morpheus discuss the nature of reality:

>Neo: This this isn't real?
>
>Morpheus: What is real? How do you define real? If you're talking about what you can feel, what you can smell, what you can taste and see, then real is silaly real in test force this realin despecial the fust of the Oracle�s apartment to inpoint. Ih ever soft that we have reached this depth.
>
>Niobe: Do you understand what you're asking?
>
>Morpheus: I am asking Now this should do the real that operation of your defense system.
>
>Morpheus: If I knew have absolutely no idea how you are able to do some of the thing: stopping you is trelans for that ship and everything that has away, or him. That's all. I juits that sample still happening hacking about we can see, Mr. Anderson.

<br>

Agent Smith interrogates Neo and then casually asks him to meet up:

>(Interrogation)
>
>Agent Smith: As you see, we�ve had our eye on you for some time now, Mr. Anderson. It seems that you've been looking for you. Now do you still want to meet?

