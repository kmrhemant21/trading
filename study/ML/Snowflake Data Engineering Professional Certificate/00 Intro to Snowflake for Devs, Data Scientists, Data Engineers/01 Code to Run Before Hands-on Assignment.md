# Code to Run Before Hands-on Assignment

## Free Snowflake Trial

We've reached an exciting moment – Now you'll get to create your Snowflake account, quickly load some data, and start doing work in Snowflake!

Signing up for a free account is very simple. As part of this course, you get a 120-day free trial account that you can access [here](). 

You should get a welcome email from Snowflake. Follow the instructions on the email to "activate" your account and create a user/password combination to access your account.

And that's it for signing up! You now have a 120-day free trial Snowflake account.

## Ingesting One Table

Next we want to ingest the data you'll be using in the following assignment. Just open up your free trial account, copy the code below, paste it into a SQL Worksheet, and run it. Be sure to 'Run All' by clicking on the down arrow next to the run button. By pressing just the run button, this will only run the line of code that the cursor is on. This will create your "tasty_bytes_sample_data.raw_pos.menu" table and load data into it. You don't need to worry about what this code is doing – we'll cover all of this later in the course.

```sql
USE ROLE accountadmin;

USE WAREHOUSE compute_wh;

---> create the Tasty Bytes Database
CREATE OR REPLACE DATABASE tasty_bytes_sample_data;

---> create the Raw POS (Point-of-Sale) Schema
CREATE OR REPLACE SCHEMA tasty_bytes_sample_data.raw_pos;

---> create the Raw Menu Table
CREATE OR REPLACE TABLE tasty_bytes_sample_data.raw_pos.menu
(
    menu_id NUMBER(19,0),
    menu_type_id NUMBER(38,0),
    menu_type VARCHAR(16777216),
    truck_brand_name VARCHAR(16777216),
    menu_item_id NUMBER(38,0),
    menu_item_name VARCHAR(16777216),
    item_category VARCHAR(16777216),
    item_subcategory VARCHAR(16777216),
    cost_of_goods_usd NUMBER(38,4),
    sale_price_usd NUMBER(38,4),
    -+-+-+-+-+
# Code to Run Before Hands-on Assignment

## Setting Up Your Free Snowflake Trial

We've reached an exciting milestone – it's time to create your Snowflake account, load sample data, and begin working with Snowflake directly!

Creating a complimentary account is straightforward. This course provides you with a 120-day trial account that you can access [here](). 

You'll receive a welcome email from Snowflake. Follow the email instructions to "activate" your account and establish your login credentials.

That completes the registration process! You now have access to a 120-day trial Snowflake environment.

## Loading Sample Data

The next step involves loading the dataset for your upcoming assignment. Simply access your trial account, copy the SQL script below, paste it into a SQL Worksheet, and execute it. Make sure to select 'Run All' using the dropdown arrow beside the run button. Clicking only the run button executes just the current line where your cursor is positioned. This script will establish your "tasty_bytes_sample_data.raw_pos.menu" table and populate it with data. Don't worry about understanding every detail of this code – we'll explore these concepts thoroughly later in the course.

```sql
USE ROLE accountadmin;

USE WAREHOUSE compute_wh;

-- Create the Tasty Bytes Database
CREATE OR REPLACE DATABASE tasty_bytes_sample_data;

-- Create the Raw POS (Point-of-Sale) Schema
CREATE OR REPLACE SCHEMA tasty_bytes_sample_data.raw_pos;

-- Create the Raw Menu Table
CREATE OR REPLACE TABLE tasty_bytes_sample_data.raw_pos.menu
(
    menu_id NUMBER(19,0),
    menu_type_id NUMBER(38,0),
    menu_type VARCHAR(16777216),
    truck_brand_name VARCHAR(16777216),
    menu_item_id NUMBER(38,0),
    menu_item_name VARCHAR(16777216),
    item_category VARCHAR(16777216),
    item_subcategory VARCHAR(16777216),
    cost_of_goods_usd NUMBER(38,4),
    sale_price_usd NUMBER(38,4),
    menu_item_health_metrics_obj VARIANT
);

-- Create the Stage referencing the Blob location and CSV File Format
CREATE OR REPLACE STAGE tasty_bytes_sample_data.public.blob_stage
url = 's3://sfquickstarts/tastybytes/'
file_format = (type = csv);

-- Query the Stage to find the Menu CSV file
LIST @tasty_bytes_sample_data.public.blob_stage/raw_pos/menu/;

-- Copy the Menu file into the Menu table
COPY INTO tasty_bytes_sample_data.raw_pos.menu
FROM @tasty_bytes_sample_data.public.blob_stage/raw_pos/menu/;
```