package wekapro;

import weka.associations.FPGrowth;
import weka.core.Instances;

public class FPGrowthModel extends KnowledgeModel {

    Instances newData;
    FPGrowth fp;
    public FPGrowthModel() {
    }

    public FPGrowthModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
        this.fp = new FPGrowth();
    }
    
    public void mineAssociationRules() throws Exception{
        //Loc du lieu
        //Remove attribute
        this.newData = removeData(this.wekaDataset);

        //convert data type
        //this.newData = convertToBinary(this.wekaDataset);

        // Thiet lap thong so cho model Apriori
        fp.setOptions(this.model_options);

        //Khai pha luat ket hop bang thuat toan Apriori
        fp.buildAssociations(this.newData);
    }

    @Override
    public String toString() {
        return fp.toString();
    }

    
}
