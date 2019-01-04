package com.neural.network.test;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by zhouyswo on 2018/10/5.
 */
public class INDArrayTest {

    public static void main(String[] args) {
        try {
//            RecordReader rr = new CSVRecordReader();
//            rr.initialize(new FileSplit(new ClassPathResource("/DataExamples/animals/animals_train.csv").getFile()));
//            DataSetIterator iterator = new RecordReaderDataSetIterator(rr, 30, 4, 3);
//            DataSet sets = iterator.next();
//            INDArray arr = sets.getFeatures();
//            //System.out.println( arr.get());
//            System.out.println( arr.get(NDArrayIndex.all(),NDArrayIndex.interval(0,2)));
            //INDArray rows = arr.getRow(0);
           // System.out.println(rows);
//            for (int i = 0; i < rows.length(); i++) {
//                System.out.println(rows.getDouble(i));
//            }
            //System.out.println(Nd4j.zeros(10,10));
            //System.out.println( Nd4j.zeros(1, 2* 28));
            //System.out.println("lllllll");
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

    }
}
