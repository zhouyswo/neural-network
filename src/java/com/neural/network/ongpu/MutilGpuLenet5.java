package com.neural.network.ongpu;

import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by DELL on 2019/1/4.
 */
public class MutilGpuLenet5 {
    public static void main(String args[]) {

    }

    public void lenet5() {
        Nd4j.setDataType(DataBuffer.Type.HALF);
        CudaEnvironment.getInstance()
                .getConfiguration()
                .allowMultiGPU(true)
                .setMaximumDeviceCache(2 * 1024 * 1024 * 1024L)
                .allowCrossDeviceAccess(true);
        //图像通道数
        int imgChannels = 1;
        //输出类别数
        int outLabels = 10;
        //每一批次处理的数据量
        int batchSize = 128;
        //迭代轮数
        int epochs = 10;
        //随机种子
        int seed = 64;
    }

}
