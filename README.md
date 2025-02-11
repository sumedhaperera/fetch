# fetch
Task 3: Training Considerations
Discuss the implications and advantages of each scenario and explain your rationale as to how
the model should be trained given the following:
1. If the entire network should be frozen.
If the entire network is frozen, training is not going to change any of the model weights and the gradients will not be updated. This is useful only if you want to use pretrained models at inference time, or to feed the output into another model as featutures. If you use untrained models the weights will be randomly assigned and will not be of much use.

2. If only the transformer backbone should be frozen.
If only the transformer backbone is frozen, one can feed in the pretrained embeddings as features to the classifier tasks. This is useful and mostly sufficient if the domain of the new model is similar to the pretrained models. If there is not much training data to fine tune the transformer model, it makes sense to freeze the model to prevent overfitting. The training also will be much faster and less resource intense. If however the domain of the pretrained model and the domain of the new model are different, then there could be potential representation mismatch and lead to low performance.

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
If one of the task-specific heads is frozen, this is helpful of one the tasks is fully trained and the other is not. Mostly it will help in the second task generalize better (transfer learning) when the training data is low.

Consider a scenario where transfer learning can be beneficial. Explain how you would approach
the transfer learning process, including:
1. The choice of a pre-trained model.
TRansfer learning is beneficial if there are already pretrained models that have already captured semantic and cotextual knowledge. The choice of the model depends on the training domain as well as the task that it is being used for. For instance we used all-MiniLM-L6-v2 as our transformer backbone to our classification tasks because it is optimized for semantic similarity. It is also more lightweight then models like BERT.
2. The layers you would freeze/unfreeze.
Either I would freeze the entire transformer backbone or freeze the layers from top down (unfreeze the bottom layers)
3. The rationale behind these choices.
If the transfomer model is trained from within the same domain there may not be a need to retrain the embeddings. Simply transfer the embeddings and use them for a subsequent task. If there needs to be fine tuning then the top layers that have more generalized knowledge should be frozen and the bottom layers unfrozen. Unfreezing the top layers migh lead to catastrophic forgetting.
