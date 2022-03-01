/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package wekapro;

import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.instance.Resample;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author KameDev
 */
public class KnowledgeModel {
    DataSource wekaSource;
    Instances wekaDataset;
    String[] model_options;
    String[] data_options;
    Instances trainSet;
    Instances testSet;

    public KnowledgeModel() {

    }

    public KnowledgeModel(String filename, String m_opts, String d_opts) throws Exception {
        if(!filename.isEmpty()){
            this.wekaSource = new DataSource(filename);
            this.wekaDataset = wekaSource.getDataSet();
        }

        if (m_opts != null) {
            this.model_options = weka.core.Utils.splitOptions(m_opts);
        }
        
        if(d_opts != null){
            this.data_options = weka.core.Utils.splitOptions(d_opts);
        }
    }

    public Instances removeData(Instances originalData) throws Exception {
        Remove remove = new Remove();
        remove.setOptions(data_options);
        remove.setInputFormat(originalData);
        return Filter.useFilter(originalData, remove);
    }

    public Instances convertData(Instances originalData) throws Exception {
        NumericToNominal n2n = new NumericToNominal();
        n2n.setOptions(data_options);
        n2n.setInputFormat(originalData);
        return Filter.useFilter(originalData, n2n);
    }

    public Instances convertToBinary(Instances originalData) throws Exception {
        NominalToBinary n2b = new NominalToBinary();
        n2b.setOptions(data_options);
        n2b.setBinaryAttributesNominal(true);
        n2b.setInputFormat(originalData);
        return Filter.useFilter(originalData, n2b);
    }

    public void saveData(String filename) throws IOException {
        ArffSaver outData = new ArffSaver();
        outData.setInstances(this.wekaDataset);
        outData.setFile(new File(filename));
        outData.writeBatch();
        System.out.println("Finished");
    }

    public void saveDataToCSV(String filename) throws IOException {
        CSVSaver outData = new CSVSaver();
        outData.setInstances(this.wekaDataset);
        outData.setFile(new File(filename));
        outData.writeBatch();
        System.out.println("Converted");
    }

    public Instances divideTrainTest(Instances originalSet, double percent, boolean isTest) throws Exception {
        RemovePercentage rp = new RemovePercentage();
        rp.setPercentage(percent);
        rp.setInvertSelection(isTest);
        rp.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rp);
    }
    
    public Instances divideTrainTestR(Instances originalSet, double percent, boolean isTest) throws Exception {
        Resample rs = new Resample();
        rs.setNoReplacement(true);
        rs.setSampleSizePercent(percent);
        rs.setInvertSelection(isTest);
        rs.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rs);
    }

    public void saveModel(String filename, Object model) throws Exception{
        weka.core.SerializationHelper.write(filename, model);
    }

    public Object loadModel(String filename) throws Exception{
        return weka.core.SerializationHelper.read(filename);
    }

    public void setTrainset(String filename) throws Exception{
        DataSource trainSource = new DataSource(filename);
        this.trainSet = trainSource.getDataSet();
    }

    public void setTestset(String filename) throws Exception{
        DataSource testSource = new DataSource(filename);
        this.testSet = testSource.getDataSet();
    }

    @Override
    public String toString() {
        return wekaDataset.toSummaryString();
    }

}
