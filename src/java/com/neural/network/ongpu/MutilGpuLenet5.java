package com.neural.network.ongpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by DELL on 2019/1/4.
 */
public class MutilGpuLenet5 {
    public static void main(String args[]){

    }
    public void lenet5(){
        Nd4j.setDataType(DataBuffer.Type.HALF);
    }

}
