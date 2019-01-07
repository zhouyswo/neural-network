package cudnn.alexnet;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by DELL on 2019/1/7.
 */
public class Alexnet {
    private long seed = 123;
    private int width = 224;
    private int hight = 224;
    private int channels = 3;
    private int classnum = 10;
    private IUpdater updater = new Nesterovs(0.01, 0.9);
    private CacheMode cmdel = CacheMode.NONE;
    private WorkspaceMode wsmodel = WorkspaceMode.ENABLED;
    private ConvolutionLayer.AlgoMode algo = ConvolutionLayer.AlgoMode.PREFER_FASTEST;


    public static void main(String args[]) {

    }

    public MultiLayerNetwork getAlexNetwork() {
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .trainingWorkspaceMode(wsmodel)
                .inferenceWorkspaceMode(wsmodel)
                .cacheMode(cmdel)
                .l2(0.00005)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(11, 11)
                        .stride(4, 4)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder().build())
                .layer(2, new Subsampling1DLayer.Builder()
                        .kernelSize(3)
                        .stride(2)
                        .padding(1)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(3, new ConvolutionLayer.Builder()
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nOut(256)
                        .biasInit(1).build())
                .layer(4, new Subsampling1DLayer.Builder()
                        .kernelSize(3)
                        .stride(2)
                        .convolutionMode(ConvolutionMode.Truncate).build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .convolutionMode(ConvolutionMode.Same)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384).build())
                .layer(7, new ConvolutionLayer.Builder()
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(256)
                        .biasInit(1).build())
                .layer(8, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(256)
                        .biasInit(1).build())
                .layer(9, new SubsamplingLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(10, new DenseLayer.Builder().nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(1)
                        .dropOut(0.5).build())
                .layer(11, new DenseLayer.Builder().nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(1)
                        .dropOut(0.5).build())
                .layer(12, new OutputLayer.Builder()
                        .nOut(classnum)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build())
                .setInputType(InputType.convolutional(hight, width, channels))
                .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(mlc);
        mln.setListeners(new ScoreIterationListener(64));
        mln.init();
        return mln;
    }
}
