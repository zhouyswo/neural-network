package multigpu.lenet5;//package com.neural.network.ongpu;

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
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.FilenameFilter;
import java.util.*;

/**
 * Created by DELL on 2019/1/4.
 */
public class MultiGpuLenet5 {
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
        new MultiGpuLenet5().trainLenet5();
    }

    public void trainLenet5() throws Exception {
        Nd4j.setDataType(DataBuffer.Type.HALF);
        CudaEnvironment.getInstance()
                .getConfiguration()
                .allowMultiGPU(true)
                .setMaximumDeviceCache(7 * 1024 * 1024 * 1024L)
                .allowCrossDeviceAccess(true);

        PathLabelGenerator lableMaker = new ParentPathLabelGenerator();

        FileSplit files = new FileSplit(new File(datapath), NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RandomPathFilter randomf = new RandomPathFilter(new Random(seed), NativeImageLoader.ALLOWED_FORMATS);

        InputSplit[] splits = files.sample(randomf, 0.7, 0.3);
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
        MultiLayerNetwork model = getLenet5Model();
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
        Evaluation eval = new Evaluation(outLabels);
        while(testdsi.hasNext()){
            DataSet ds = testdsi.next();
            INDArray output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }
        System.out.println(eval.stats());
        testdsi.reset();
    }

    public MultiLayerNetwork getLenet5Model() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed).l2(0.0005).weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs.Builder().learningRate(0.01).build())
                .biasUpdater(new Nesterovs.Builder().learningRate(0.01).build())
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(imgChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(2, new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(5).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outLabels).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(hight, width, imgChannels))
                .backpropType(BackpropType.Standard)
                .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setListeners(new ScoreIterationListener(100));
        return mln;
    }


}
