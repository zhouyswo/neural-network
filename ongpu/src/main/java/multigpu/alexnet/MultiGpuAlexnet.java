package multigpu.alexnet;

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
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.io.File;
import java.io.FilenameFilter;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Created by DELL on 2019/1/4.
 */
public class MultiGpuAlexnet {
    //图像通道数
    int imgChannels = 3;
    int width = 224;
    int hight = 224;
    //输出类别数
    int outLabels = 5;
    //每一批次处理的数据量
    int batchSize = 128;
    //迭代轮数
    int epochs = 5;
    //随机种子
    int seed = 64;
    String datapath = "/opt/workplace/testDatas/data/flower_photos";

    public static void main(String args[]) throws Exception {
        new MultiGpuAlexnet().trainAndTestAlexnet();
    }
    public void trainAndTestAlexnet() throws  Exception{
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

        ImageRecordReader trainirr = new ImageRecordReader(width, hight, imgChannels, lableMaker,itf);
        //trainirr.setLabels(Arrays.asList(dierctories));
        trainirr.initialize(train);

        ImageRecordReader testirr = new ImageRecordReader(width, hight, imgChannels, lableMaker,itf);
        // testirr.setLabels(Arrays.asList(dierctories));
        testirr.initialize(test);

        DataSetIterator traindsi = new RecordReaderDataSetIterator(trainirr, batchSize, 1, 5);
        DataSetIterator testdsi = new RecordReaderDataSetIterator(testirr, batchSize, 1, 5);

        //预处理器
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(traindsi);
        traindsi.setPreProcessor(scaler);
        testdsi.setPreProcessor(scaler);
        MultiLayerNetwork model = configAlexnet();
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
    public MultiLayerNetwork configAlexnet(){
//        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
//                .l2(0.0005)
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
//                .list()
//                .layer(0,new ConvolutionLayer.Builder().kernelSize(11,11).stride(4,4).padding(3,3).nIn(imgChannels).nOut(96).biasInit(0).activation(Activation.RELU).name("conv1").build())
//                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
//                .layer(2,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).build())
//                .layer(3,new ConvolutionLayer.Builder().kernelSize(5,5).stride(1,1).padding(2,2).nOut(256).biasInit(1).activation(Activation.RELU).build())
//                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
//                .layer(5,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).build())
//                .layer(6,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).nOut(384).biasInit(0).activation(Activation.RELU).build())
//                .layer(7,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).nOut(384).biasInit(1).activation(Activation.RELU).build())
//                .layer(8,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).nOut(256).biasInit(1).activation(Activation.RELU).build())
//                .layer(9,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).build())
//                .layer(10,new DenseLayer.Builder().dropOut(0.5).nOut(4096).biasInit(1).dist( new GaussianDistribution(0, 0.005)).build())
//                .layer(11,new DenseLayer.Builder().dropOut(0.5).nOut(4096).biasInit(1).dist( new GaussianDistribution(0, 0.005)).build())
//                .layer(12,new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation(Activation.SOFTMAX).build())
//                .setInputType(InputType.convolutional(hight, width, imgChannels)
//                ).backpropType(BackpropType.Standard)
//                .pretrain(true)
//                .build();
//        MultiLayerNetwork mln = new MultiLayerNetwork(mlc);
//        mln.setListeners(new ScoreIterationListener(batchSize));
//        mln.init();
//        return mln;

        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .list()
                .layer(0,new ConvolutionLayer.Builder().kernelSize(11,11).stride(4,4).padding(3,3).nIn(imgChannels).nOut(48).biasInit(0).activation(Activation.RELU).name("conv1").build())
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).build())
                .layer(3,new ConvolutionLayer.Builder().kernelSize(5,5).stride(1,1).padding(2,2).nOut(96).biasInit(1).activation(Activation.RELU).build())
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).build())
                .layer(6,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).nOut(160).biasInit(0).activation(Activation.RELU).build())
                .layer(7,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).nOut(160).biasInit(1).activation(Activation.RELU).build())
                .layer(8,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).nOut(96).biasInit(1).activation(Activation.RELU).build())
                .layer(9,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).build())
                .layer(10,new DenseLayer.Builder().dropOut(0.5).nOut(512).biasInit(1).dist( new GaussianDistribution(0, 0.005)).build())
                .layer(11,new DenseLayer.Builder().dropOut(0.5).nOut(512).biasInit(1).dist( new GaussianDistribution(0, 0.005)).build())
                .layer(12,new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(hight, width, imgChannels)
                ).backpropType(BackpropType.Standard)
                .pretrain(true)
                .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(mlc);
        mln.setListeners(new ScoreIterationListener(batchSize));
        mln.init();
        return mln;
    }
}
