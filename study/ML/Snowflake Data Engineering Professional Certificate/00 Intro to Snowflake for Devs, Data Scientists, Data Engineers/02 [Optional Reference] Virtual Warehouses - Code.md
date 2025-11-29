# [Optional Reference] Virtual Warehouses - Code

This is not mandatory reading, but here's the code we'll run in the "Virtual Warehouses" videos. It may come in handy when you're doing the associated hands-on assignment.

## Query Examples

**What menu items does the Freezing Point brand sell?**
```sql
SELECT 
    menu_item_name
FROM tasty_bytes_sample_data.raw_pos.menu
WHERE truck_brand_name = 'Freezing Point';
```

**What is the profit on Mango Sticky Rice?**
```sql
SELECT 
    menu_item_name,
    (sale_price_usd - cost_of_goods_usd) AS profit_usd
FROM tasty_bytes_sample_data.raw_pos.menu
WHERE 1=1
AND truck_brand_name = 'Freezing Point'
AND menu_item_name = 'Mango Sticky Rice';
```

## Warehouse Management

**Create warehouses:**
```sql
CREATE WAREHOUSE warehouse_dash;
CREATE WAREHOUSE warehouse_gilberto;
```

**Show all warehouses:**
```sql
SHOW WAREHOUSES;
```

**Use a specific warehouse:**
```sql
USE WAREHOUSE warehouse_gilberto;
```

**Set warehouse size to medium:**
```sql
ALTER WAREHOUSE warehouse_dash SET warehouse_size=MEDIUM;
```

**Switch to another warehouse:**
```sql
USE WAREHOUSE warehouse_dash;
```

**Query with profit calculation and ordering:**
```sql
SELECT
     menu_item_name,
    (sale_price_usd - cost_of_goods_usd) AS profit_usd
FROM tasty_bytes_sample_data.raw_pos.menu
ORDER BY 2 DESC;
```

**Set warehouse size to xsmall:**
```sql
ALTER WAREHOUSE warehouse_dash SET warehouse_size=XSMALL;
```

**Drop a warehouse:**
```sql
DROP WAREHOUSE warehouse_vino;
```

**Show warehouses after changes:**
```sql
SHOW WAREHOUSES;
```

**Create a multi-cluster warehouse (max clusters = 3):**
```sql
CREATE WAREHOUSE warehouse_vino MAX_CLUSTER_COUNT = 3;
```

**Show warehouses again:**
```sql
SHOW WAREHOUSES;
```

**Set the auto_suspend and auto_resume parameters:**
```sql
ALTER WAREHOUSE warehouse_dash SET AUTO_SUSPEND = 180 AUTO_RESUME = FALSE;
```

**Final warehouse check:**
```sql
SHOW WAREHOUSES;
```
