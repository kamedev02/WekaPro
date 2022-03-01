package wekapro;

import weka.associations.Apriori;
import weka.core.Instances;

public class AprioriModel extends KnowledgeModel {
    Apriori apriori;
    Instances newData;

    public AprioriModel() {

    }

    public AprioriModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
        this.apriori = new Apriori();
    }

    public void mineAssociationRules() throws Exception{
        //Loc du lieu
        //Remove attribute
        this.newData = removeData(this.wekaDataset);

        //convert data type
        //this.newData = convertData(this.wekaDataset);

        // Thiet lap thong so cho model Apriori
        apriori.setOptions(this.model_options);

        //Khai pha luat ket hop bang thuat toan Apriori
        apriori.buildAssociations(this.newData);
    }

    @Override
    public String toString() {
        return apriori.toString();
    }

    
}
