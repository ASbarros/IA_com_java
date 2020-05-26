package anderson;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuiler;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.UseNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PersonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class RecommenderBuiler implements RecommenderBuiler {

  public Recommender buildRecommender(DataModel model) throws TasteException {
      UserSimilarity similarity = new PersonCorrelationSimilarity(model);
      UseNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
      UserBasedRecommender recommender= new GenericUserBasedRecommender(model, neighborhood, similarity);
      return recommender;
  }
}
