# Electric consumption prediction. 

This project is an **energy disaggregation problem** on a **time series**, generally known as NILM. From a noisy signal, a household's overall consumption over time, **we aim to predict individual consumption of four appliances, a washing machine, a fridge-freezer, a TV and a kettle, 'non-intrusively'**.
  
Each appliance has a **distinct signature**. This signature depends on user consumption patterns (washing-machine on week-ends for instance), product design (fridge-freezer is nearly always on), and the power each appliance consumes when on (kettle presents high spikes for example). In the **data exploration** phase, we studied the behavior of our appliances  compared to the total electrical consumption of the household on different timescales. It appears that individual appliances bear different relations to consumption depending on whether they are **on or off**, showing that detecting **edges and spikes**  in consumption could be very useful for predictions.   
   
For **feature engineering** and **modelling**, we tested two main approaches. 
- The first was to **engineer features manually to feed into our XGBoost**.  
 We came up with a set of overall good features for a MultiOutput XGBoost, and then improved our predictions by tweaking them into a **relevant set by appliance**, focusing on **fridge** especially as its weight in the custom metric was high. Our features can be regrouped in three categories: weather (temperature, wind and pressure were the ones we kept based on literature research), binary-encoded periodic information (is_teatime, is_winter...), and **consumption transformations** (past and future lags, moving means & std, cumulated sums, etc). We tested each feature to have an idea of relevance, using feature importances of a random forest as an initial subset selector, to then refine. 
- The second was to test a **bidirectional LSTM neural network** based on **consumption only** to predict fridge-freezer, using the results of paper "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation." by J. Kelly, W Knottenbelt et al. The rationale is that a neural network, if well designed, would learn relevant features and dependencies itself.  
You will find below our observations, the functions to code some of the features we thought of, and our model. In an appendix, you will find experiments we led and finally discarded.

On the ENS data challenge platform, our team is called **vicdetermont & naomiserf & sjallot**.
