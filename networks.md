# Networks

Define some common neural networks architectures and ideas.

## Computer vision

### Image classification

**Base structure**: Convolution => Activation => Pooling (AlexNet like)

**Network in Network**: Use 1x1 convolutions before the convolutions (act as fully connected layers)

**Inception**: Concatenate filters from different sizes together. Use 1x1 convolutions before and after to reduce/restore the dimensions.

**ResNet**: Use residual connections by connecting the input to the output (`y = conv(x) + x`). Between the layers of different size (for instance after pooling), use 1x1 convolution (to adapt the depth) with striding (to reduce the h/w) on the residual connection (`y = conv(x) + conv1x1(x)`).

**Binary network**: The weights of each layers are either +1/-1 (multiply by a floating constant and a bias different for each layer). During forward pass, the network is binarized. During the backward pass, the weights are updated as float. A version exist which also binarize the input (XNOR network).

### Detection

**Before R-CNN**: Sliding windows to classify at each positions.

**YOLO**: Divide the image in a grid of cell. Each cell will predict multiple bounding box candidates with a confidence score (P(Obj)) and each cell predict which object would be in the cell if there was one (ex: P(Cars|Obj)). The bounding box are then thresholded using the confidence score. Each one of the w*h cells predict a vector `[[centerx, centery, w, h, P(obj)]&ast;nb_of_proposal, [P(Car|Obj),..., P(Pers|Obj)]]`.

**SSD**: Region proposal (bounding boxes) to segment object, then classification, then overlapping detection.


### Segmentation

**Multi-task Network Cascades**: Based on ResNet to perform instance based segmentation. Use cascade loss function to divide the segmentation task into 3 sub-tasks. Each task uses as input the output of the previous one (in addition to the shared features computed by the CNN).<br />
*Instance-aware Semantic Segmentation via Multi-task Network Cascades, Jifeng Dai Kaiming He et al.*

**DeepMask / Multipath**: 2 networks on to segment objects independently of the class and one to give a label to the segmentation.

**Aerial Scenes Segmentation**: Data quality is important: instead of using binary mask (presence or not of the object) as ground truth, weight the mask (each pixel is weighted by the closed distance to the boundaries). Also bilinear up-sampling of the features maps (due to low resolution of the image (object to detect really small)), feed that to a FC to segment each pixel.
*Automatic Building Extraction in Aerial Scenes Using Convolutional Networks, Jiangye Yuan*, ([Arxiv](https://arxiv.org/abs/1602.06564))

### Image Captioning

**Show and Tell**: Use a RNN to generate a sentence using as input the feature map computed by a CNN.

### Face Recognition

**FaceNet**: Use CNN to project the face in a 128-dimensional space (on an hypersphere). Trained with triplet embedding (triplet(anchor, pos, neg)). Try to conjointly minimize dist(anchor, pos) while maximizing dist(anchor, neg).

### Other

**Neural Style**: Learn the input from white noise (the network has fixed weight and is a CNN trained on ImageNet). Isolate style and content. The loss function has two term. Style matching using Gram Matrix (capture the correlations between filters). Content matching: activations have to match the target image (same content).

**Image search/retrival**: Project the image into an embedding space. Search close match using KNN with the previously indexed images. Approximate KNN with KD-Tree.

**Super resolution images**:...

**Image compression**:...

## Natural Language Processing

**RTNN**: Recursive neural tensor network (not recurrent). Tree-like shape. Use 3d tensor in addition to matrix as weights. Original model used for sentiment analysis.

**Word2Vec**: Project each word in a high dimensional space which encode its semantic meaning (embedding).

**seq2seq**: 2 RNN. The encoder compute a though vector encoding the sentence meaning, the decoder.

## Reinforcement learning

**Deep Q-Network**: Use a CNN to learn Q(s,a).

**Double Q-Learning**:

**A3C**:

**UNREAL**: Based on A3C, augment the cost function by adding auxiliary tasks.

**Neural Architecture Search**: Generate new networks architecture by formalizing a network architecture as a sequence and training a RNN to generate it using REINFORCE.

* For CNN, the network sequentially generate filter height, stride, anchors,... for each layers. The anchor allows the connect the layer to a previous one to add skip connections to the network.
* A version allows to generate RNN cells by formalising a RNN cell as a tree and sequentially generating the nodes properties.

*Neural Architecture Search with Reinforcement learning, Barret Zoph, Quoc V. Le* ([Arxiv](https://arxiv.org/abs/1611.01578))

## Other (low level)

**Deep learning on graph**: Generalization of convolution to sparse data (organized as a graph). Based on the field of signal processing on graph which define operations like the Fourier transform for graphs.

**LSTM Variant**: GPU, Grid-LSTM, Bidirectional-LSTM,...

**Attention mechanism**:

**Memory networks**:

**VAE**:

**Draw**:

**Pixel**:

**PixelCNN**:

**WaveNet**:

...
