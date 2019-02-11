package multigpu.vgg;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.FilenameFilter;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Created by DELL on 2019/1/4.
 */
public class MultiGpuVGG16A {
    private int width = 224;
    private int hight = 224;
    private int channels = 3;
    private int epochs = 5;
    private int betchsize = 128;
    private int labes = 5;
    private int seed = 32;
    String datapath = "/opt/workplace/testDatas/data/flower_photos";   //caltech101_ObjectCategories";

    public static void main(String args[]) {

    }
    public void trainAndTestVgg() throws  Exception{
        Nd4j.setDataType(DataBuffer.Type.HALF);
        CudaEnvironment.getInstance()
                .getConfiguration()
                .allowMultiGPU(true)
                .setMaximumDeviceCache(7 * 1024 * 1024 * 1024L)
                .allowCrossDeviceAccess(true);

        PathLabelGenerator lableMaker = new ParentPathLabelGenerator();

        FileSplit files = new FileSplit(new File(datapath), NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RandomPathFilter randomf = new RandomPathFilter(new Random(seed), NativeImageLoader.ALLOWED_FORMATS);

        InputSplit[] splits = files.sample(randomf, 0.9, 0.1);
        InputSplit train = splits[0];
        InputSplit test = splits[1];

        //搜索文件目录
        String[] dierctories = new File(datapath).list(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return new File(dir, name).isDirectory();
            }
        });
        List<Pair<ImageTransform, Double>> pipeline = new LinkedList<Pair<ImageTransform, Double>>();
        pipeline.add(new Pair<ImageTransform, Double>(new ResizeImageTransform(width, hight), 1.0));
        pipeline.add(new Pair<ImageTransform, Double>(new FlipImageTransform(1), 0.5));

        ImageTransform itf = new PipelineImageTransform(pipeline, false);

        ImageRecordReader trainirr = new ImageRecordReader(width, hight, channels, lableMaker,itf);
        //trainirr.setLabels(Arrays.asList(dierctories));
        trainirr.initialize(train);

        ImageRecordReader testirr = new ImageRecordReader(width, hight, channels, lableMaker,itf);
        // testirr.setLabels(Arrays.asList(dierctories));
        testirr.initialize(test);

        DataSetIterator traindsi = new RecordReaderDataSetIterator(trainirr, betchsize, 1, 5);
        DataSetIterator testdsi = new RecordReaderDataSetIterator(testirr, betchsize, 1, 5);

        //预处理器
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(traindsi);
        traindsi.setPreProcessor(scaler);
        testdsi.setPreProcessor(scaler);
        MultiLayerNetwork model = getVGG16Network();
        ParallelWrapper pw = new ParallelWrapper.Builder(model)
                .prefetchBuffer(32)
                .workers(3)
                .averagingFrequency(9)
                .reportScoreAfterAveraging(true)
                .build();

        long timeX = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            long time1 = System.currentTimeMillis();
            pw.fit(traindsi);
            long time2 = System.currentTimeMillis();
            System.out.println("*** Completed epoch " + i + ", time: " + (time2 - time1) + " *********");
        }
        scaler.fit(testdsi);
        Evaluation eval = model.evaluate(testdsi);
        System.out.println(eval.stats(true));
    }

    public MultiLayerNetwork getVGG16Network() throws Exception {
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
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(10, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(11, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(12, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(13, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(14, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(15, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(16, new Convolution1DLayer.Builder().kernelSize(3, 3).stride(1).padding(1).activation(Activation.RELU).nOut(512).build())
                .layer(17, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(18, new DenseLayer.Builder().nOut(4096).activation(Activation.RELU).dropOut(0.5).build())
                .layer(19, new DenseLayer.Builder().nOut(4096).activation(Activation.RELU).dropOut(0.5).build())
                .layer(20, new DenseLayer.Builder().nOut(1000).activation(Activation.RELU).build())
                .layer(21, new OutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).nOut(5).build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(hight,width,channels))
                .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(mlc);
        mln.setListeners(new ScoreIterationListener(betchsize));
        mln.init();
        return mln;
    }
}
