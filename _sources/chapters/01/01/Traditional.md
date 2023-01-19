## The traditional M&V approach

The most common approach to the measurement and verification (M&V) of the energy savings from a retrofit is to treat M&V as a *prediction* task. In this case, a predictive model is developed using the pre-retrofit data to predict the building’s energy consumption given the values of a set of observable variables. In most cases, these variables correspond to calendar features (as a proxy for the operation schedule), such as the week of the year and the hour of the week, and outdoor temperature information. After the energy retrofit, this model is used to predict the counterfactual consumption. The difference between the counterfactual and the actual consumption is regarded as avoided energy usage that can be attributed to the intervention. 

The two main assumptions behind this approach is that:

1. All the aspects of the building's operation that are approximated through the calendar features (first and foremost, the operating schedule) remain the same during the post-intervention period. 

2. The characteristics – in the sense of the conditional distributions – of all the unobserved variables that have not been used for the training of the predictive model also remain the same during the post-intervention period. 

This way of defining M&V is summarised in the next diagram:

![Predict](../../../images/predictive.png)

The problem with this approach is that a prediction task ends the moment we get the [ground truth](https://en.wikipedia.org/wiki/Ground_truth) data. We predicted something, we see the actual outcome, and we evaluate how successful the prediction was. However, M&V is an *impact assessment* task, so there is no reason we cannot revise both our predictions and our models as soon as we get data from the post-intervention period.