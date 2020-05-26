package anderson;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.IOException;
import java.util.List;

public class RecomendaFilmes {
    public static void main(String[] args) throws IOException, TasteException {
        DataModel filmes = new Recomendados().getModeloDeFilmes();
        Recommender recommender = new RecomendadorBuilder().buildRecommender(filmes);
        
        List<RecommendedItem> recommendations = recommender.recommend(4, 3);
        for (RecommendedItem recommendation : recommendations) {
            System.out.println("Voce pode gostar deste filme");
            System.out.println(recommendation);
        }


    }
}