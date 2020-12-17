# Electric consumption prediction. 

This project is an **energy disaggregation problem** on a **time series**, generally known as NILM. From a noisy signal, a household's overall consumption over time, **we aim to predict individual consumption of four appliances, a washing machine, a fridge-freezer, a TV and a kettle, 'non-intrusively'**.
  
Each appliance has a **distinct signature**. This signature depends on user consumption patterns (washing-machine on week-ends for instance), product design (fridge-freezer is nearly always on), and the power each appliance consumes when on (kettle presents high spikes for example). In the **data exploration** phase, we studied the behavior of our appliances  compared to the total electrical consumption of the household on different timescales. It appears that individual appliances bear different relations to consumption depending on whether they are **on or off**, showing that detecting **edges and spikes**  in consumption could be very useful for predictions.   
   
For **feature engineering** and **modelling**, we tested two main approaches. 
- The first was to **engineer features manually to feed into our XGBoost**.  
 We came up with a set of overall good features for a MultiOutput XGBoost, and then improved our predictions by tweaking them into a **relevant set by appliance**, focusing on **fridge** especially as its weight in the custom metric was high. Our features can be regrouped in three categories: weather (temperature, wind and pressure were the ones we kept based on literature research), binary-encoded periodic information (is_teatime, is_winter...), and **consumption transformations** (past and future lags, moving means & std, cumulated sums, etc). We tested each feature to have an idea of relevance, using feature importances of a random forest as an initial subset selector, to then refine. 
- The second was to test a **bidirectional LSTM neural network** based on **consumption only** to predict fridge-freezer, using the results of paper "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation." by J. Kelly, W Knottenbelt et al. The rationale is that a neural network, if well designed, would learn relevant features and dependencies itself.  

For feature engineering, here is how we proceeded. 
We focused on three variable types: **weather, periodical indicators per appliance, and consumption-derived features**.   

We manually selected **temperature, pressure and wind** out of all weather variables because literature research informed us they were the best predictors of individual consumption.  

Using timestamps, we derived minute, hour, month, day of week easily, and built **new periodical features**: salient periods by appliance (breakfast and teatime for kettle for instance) and season variables (holiday, week-end, winter...) by segmenting the time space. We encoded some of our indicators in cosine and sine format to identify recurring trends (day, hour, day of week). 

Our most useful features overall came from **consumption transformations**.  
From literature research, we first thought of lags and rolling consumption features, both in past and future (expanding consumption didn't prove useful). We then found that normalising consumption could be useful: hence features like consumption minus rolling mean on different periods. Extracting detrended consumption from the seasonal API didn't prove useful (but we did it!). We then thought of using cumulative sums of our normalised consumption features, and differences between steps before and after a given observation. To choose our windows, we grounded our reasoning on both our data exploration (fridge has spikes within 5 minutes, so rolling std would be useful on that window) and literature research (10, 30 minutes are usually good timeframes for fridge as per *Profiling Household Appliance Electricity Usage with N-Gram Language Modeling*, Daoyuan et al). 


On the ENS data challenge platform, our team is called **vicdetermont & naomiserf & sjallot**.
