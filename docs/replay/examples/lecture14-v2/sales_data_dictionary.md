# Constructed sales data dictionary

Original synthetic data, seed 4031. Each row represents one independent fictional store and its weekly sales. There are no repeated stores or actual collected measurements. Predictor values describe that store's constructed week and market. Indices are synthetic numerical scores, not validated instruments. This example describes associations; it provides no causal evidence.

| Column | Definition and units |
|---|---|
| store_id | Unique fictional store identifier; exclude from predictors |
| digital_ad_spend | Weekly digital advertising spending, dollars |
| print_ad_spend | Weekly print advertising spending, dollars |
| promotion_index | Synthetic promotion intensity, index points |
| floor_area | Store floor area, square feet |
| staff_hours | Total weekly staffing, person-hours |
| shelf_capacity_index | Synthetic shelf capacity, index points |
| local_demand_index | Synthetic local demand, index points |
| footfall_index | Synthetic visitor activity, index points |
| neighborhood_activity_index | Synthetic neighborhood activity, index points |
| store_code_index | Arbitrary synthetic candidate input, 0 to 100 |
| display_color_index | Arbitrary synthetic candidate input, 0 to 100 |
| survey_noise_index | Arbitrary synthetic candidate input, 0 to 100 |
| weekly_sales | Prediction target: weekly sales revenue, dollars |

The marketing group contains digital spending, print spending, and promotion intensity. The size group contains floor area, staff hours, and shelf capacity. The demand group contains local demand, footfall, and neighborhood activity. Each group shares a normally distributed latent factor. Indices in these groups are centered at 50 and are not constrained to [0,100]. The three arbitrary candidate inputs are independent of the generating factors and have no direct role in sales generation.

The exact equation and random distributions are saved in sales_generation_metadata.json and implemented in generate_sales_data.py. The independent sales noise has standard deviation $3,000. A target-independent split reserves 72 rows for a single final comparison; all initial plots and summaries use the other 168 rows. The full CSV is saved for reproducibility, but reserved outcomes are not summarized or used for development.
