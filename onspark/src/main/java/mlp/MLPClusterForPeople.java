package mlp;

import com.beust.jcommander.JCommander;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhouyswo on 2018/10/5.
 * 多层感知机
 */
public class MLPClusterForPeople {
    private static final Logger log = LoggerFactory.getLogger(MLPClusterForPeople.class);
    private static int batchSizePerWorker = 16;
    private static int numEpochs = 2;

    public static void main(String[] args) throws  Exception {
            SparkConf spconf = new SparkConf();
            spconf.setAppName("neural-network");
            spconf.setMaster("local[24]");
           // spconf.set("spark.cores.max","");
            spconf.set("spark.executor.memory","6g");
            spconf.set("spark.executor.instances","6");
            spconf.set("spark.executor.cores","4");
//           // spconf.setExecutorEnv("instances","6");
            //spconf.setExecutorEnv("memory","6g");
            //spconf.setExecutorEnv("cores","4");
            JavaSparkContext sc = new JavaSparkContext(spconf);
            JCommander jc = new JCommander();
            RecordReader rr = new CSVRecordReader();
            File file = new File("/opt/workplace/testDatas/data/people.txt");
           // System.out.println(file.exists());
            rr.initialize(new FileSplit(file));

            DataSetIterator iterTrain = new RecordReaderDataSetIterator(rr, 10,0, 13);
//            //System.out.println(iterTrain.next().get(0));

            //一次next只拿batchSizePerWorker条数据
            //System.out.println(iterTrain.next().numExamples()+"   "+iterTest.next().numExamples());
            List<DataSet> trainDataList = new ArrayList<DataSet>();
            List<DataSet> testDataList = new ArrayList<DataSet>();
            int n = 0;
            while(iterTrain.hasNext()){
                DataSet ds = iterTrain.next();
                //System.out.println(ds.getLabels());
                //System.out.println(ds.getFeatures());
                trainDataList.add(ds);
                if(n%3==0){
                    testDataList.add(ds);
                }
               // System.out.println(n);
                n++;
            }

            JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
            JavaRDD<DataSet> testData = sc.parallelize(testDataList);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(3)
                    .activation(Activation.LEAKYRELU)//激活函数
                    .weightInit(WeightInit.XAVIER)//权重初始化
                    .updater(new Nesterovs(0.1))//更新器
                    .l2(1e-4)//正则化,有助于避免在训练时发生过拟合,L2正则化的常用值为1e-3到1e-6。
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(8).nOut(40).build())
                    //.layer(1, new DenseLayer.Builder().nIn(15).nOut(30).build())
                    .layer(1, new DenseLayer.Builder().nIn(40).nOut(30).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(30).nOut(13).build())
                    .pretrain(true)
                    .backprop(true)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(10)
                    .averagingFrequency(10)//控制参数平均化和再分发的频率，按大小等于batchSizePerWorker的微批次的数量计算
                    .workerPrefetchNumBatches(2)//Spark工作器能够以异步方式预抓取多少个微批次
                    .batchSizePerWorker(batchSizePerWorker)//控制每个工作器的微批次大小
                    .build();
            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
            for (int i = 0; i < numEpochs; i++) {
                sparkNet.fit(trainData);
                System.out.println("Completed Epoch {}"+i);
            }
            Evaluation evaluation = sparkNet.doEvaluation(testData, 10, new Evaluation(13))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
            System.out.println("***** Evaluation *****");
            System.out.println(evaluation.stats());

            //保存模型
//            File locationToSave = new File("D:/IDEAWorkPlace/neural-network/model/MLP/MLPNetWork.zip");
//            boolean saveUpdater = true;
//            ModelSerializer.writeModel(net, locationToSave, saveUpdater);
//            //读取模型
//            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
//
//            //Delete the temp training files, now that we are done with them
//            tm.deleteTempFiles(sc);


    }
}
