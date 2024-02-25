BATCH_INFERENCE_QUERY="""SELECT
    o.orderkey,
    COUNT(l.linenumber) AS lineitem_count,
    SUM(l.quantity) AS total_quantity,
    AVG(l.discount) AS avg_discount,
    SUM(l.extendedprice) AS total_extended_price,
    o.orderdate,
    o.orderpriority,
    CASE
        WHEN SUM(l.extendedprice) > 20000 THEN 1
        ELSE 0
    END AS high_value_order
FROM
    orders o
JOIN
    lineitem l ON o.orderkey = l.orderkey
GROUP BY
    o.orderkey,
    o.orderdate,
    o.orderpriority
ORDER BY 
    RANDOM() 
LIMIT 5000
"""