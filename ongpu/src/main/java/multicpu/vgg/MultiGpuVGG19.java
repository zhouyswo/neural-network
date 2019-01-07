package multicpu.vgg;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by DELL on 2019/1/4.
 */
public class MultiGpuVGG19 {
    private int width = 224;
    private int hight = 224;
    private int channels = 3;
    private int epochs = 5;
    private int betchsize = 128;
    private int labes = 5;
    private int seed = 32;

    public static void main(String args[]) {

    }

    public MultiLayerNetwork getVGG19Network() throws Exception {
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(64).build())
                .layer(1, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(64).build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(3, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(128).build())
                .layer(4, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(128).build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(6, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(256).build())
                .layer(7, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(256).build())
                .layer(8, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(256).build())
                .layer(9, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(256).build())
                .layer(10, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(11, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(12, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(13, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(14, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(15, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(16, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(17, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(18, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(19, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(20, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(21, new DenseLayer.Builder().nOut(4096).activation(Activation.RELU).dropOut(0.5).build())
                .layer(22, new DenseLayer.Builder().nOut(4096).activation(Activation.RELU).dropOut(0.5).build())
                .layer(23, new DenseLayer.Builder().nOut(1000).activation(Activation.RELU).build())
                .layer(24, new OutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).nOut(1000).build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(hight,width,channels))
                .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(mlc);
        mln.setListeners(new ScoreIterationListener(betchsize));
        mln.init();
        return mln;
    }
}
