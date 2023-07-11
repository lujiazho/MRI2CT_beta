The invoking order while training

```
training_step(ddpm.DDPM) -> shared_step(ddpm.LatentDiffusion) -> get_input(cldm.ControlLDM) -> forward(ddpm.LatentDiffusion) -> p_losses(ddpm.LatentDiffusion) -> apply_model(cldm.ControlLDM) -> 
```

---

PyTorch Lightning knows how to train the model by following a specific set of steps and guidelines. When you create a PyTorch Lightning model, you must define several methods that tell PyTorch Lightning how to train your model. Although these methods are not explicitly shown in the code snippet you provided, they would have been defined in the creation of the model itself (likely within the `create_model` function).

Here are some of the methods you typically need to define:

1. **`forward` method:** This is the method that makes a forward pass through your model, i.e., it accepts the inputs and returns the model's outputs.

2. **`training_step` method:** This is where you define the logic for a single training step. This includes how to compute the outputs (usually by calling the `forward` method) and how to calculate the loss from those outputs.

3. **`configure_optimizers` method:** This is where you define the optimizers and, optionally, learning rate schedulers. PyTorch Lightning uses the optimizers returned by this method to update your model's parameters.

4. Optionally, you can also define methods like `validation_step` and `test_step` for validation and testing, `training_epoch_end` and `validation_epoch_end` for operations at the end of each epoch, etc.

When you call `trainer.fit(model, dataloader)`, PyTorch Lightning uses the methods defined in your model to train it:

1. It fetches a batch of data from the DataLoader.
2. It feeds this batch into the `training_step` method.
3. The `training_step` method computes the outputs and loss.
4. PyTorch Lightning automatically computes the gradients of the loss with respect to the model parameters and uses the optimizers to update these parameters.
5. It repeats this process for all batches in your dataset, for as many epochs as you specified when setting up the `Trainer`.

PyTorch Lightning takes care of the details and provides a unified, simplified interface for these operations, reducing the amount of boilerplate code you need to write.