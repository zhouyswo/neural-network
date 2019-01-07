package graph.base;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BaseComputationGraph {
    public static void main(String args[]){
        new BaseComputationGraph().twoLayerNetwork();

    }

    /**
     *包含跳跃连接的循环网络
     * */
    public ComputationGraph twoLayerNetwork(){
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(0.01))
                .graphBuilder()
                .addInputs("input")
                .addLayer("ly1",new GravesLSTM.Builder().nIn(5).nOut(5).build(),"input")
                .addLayer("ly2",new RnnOutputLayer.Builder().nIn(5+5).nOut(5).build(),"input","ly1")
                .setOutputs("ly2")
                .build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        return  cg;
    }

    /**
     * 多项输入与合并
     *其中的合并点将L1和L2层输出的激活值合并（或称连接，concatenate）起来：
     *  所以，如果L1和L2层都有4个激活值输出（.nOut(4)），那么合并点的输出大小为4+4=8个激活值。
     * */
    public ComputationGraph manyInputCombineNetwork(){
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(0.01))
                .graphBuilder()
                .addInputs("input1","input2")
                .addLayer("ly1",new DenseLayer.Builder().nIn(3).nOut(4).build(),"input1")
                .addLayer("ly2",new DenseLayer.Builder().nIn(3).nOut(4).build(),"input2")
                .addVertex("merge",new MergeVertex(),"ly1","ly2")
                .addLayer("out",new OutputLayer.Builder().nIn(8).nOut(3).build(),"merge")
                .setOutputs("out")
                .build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        return  cg;
    }

    /**
     * 多任务学习
     * 多任务学习指让神经网络进行多项彼此独立的预测。 比如，想象一个同时用于分类和回归的简单网络。假设有两个输出层，“out1”用于分类，“out2”用于回归。
     * */
    public ComputationGraph multiLearnWorks(){
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(0.01))
                .graphBuilder()
                .addInputs("input")
                .addLayer("ly1",new DenseLayer.Builder().nIn(3).nOut(4).build(),"input")
                .addLayer("out1",new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4)
                        .nOut(3)
                        .build(),"ly1")
                .addLayer("out2",new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(4)
                        .nOut(2)
                        .build(),"ly1")
                .setOutputs("out1","out2")
                .build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        return  cg;
    }
}
