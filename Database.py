from sqlalchemy import create_engine, text

class Database:
    def __init__(self):
        self.engine = create_engine("postgresql://postgres:asd123*@localhost/postgres")

    def get_query1(self):
        with self.engine.connect() as conn:
            query = text("""SELECT
    c.customer_id,
    o.order_id,
    o.order_date,

    -- Siparişteki toplam harcama (tüm ürünler dahil)
    SUM(od.unit_price * od.quantity * (1 - od.discount)) AS net_spent,

    -- Siparişteki toplam ürün miktarı
    SUM(od.quantity) AS total_quantity,

    -- Siparişteki ortalama indirim oranı
    AVG(od.discount) AS avg_discount,

    -- Siparişteki ortalama ürün birim fiyatı
    AVG(od.unit_price) AS avg_unit_price,

    -- Müşteri bazlı toplam sipariş sayısı
    COUNT(DISTINCT o2.order_id) AS total_orders,

    -- Müşterinin en son sipariş tarihi
    MAX(o2.order_date) AS recent_order,

    -- Siparişin hangi çeyrekte olduğu
    EXTRACT(YEAR FROM o.order_date)::TEXT || 'Q' || EXTRACT(QUARTER FROM o.order_date)::TEXT AS year_quarter,

    -- Müşteri konum bilgileri
    c.country,
    c.city

FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_details od ON o.order_id = od.order_id

-- recent_order ve total_orders hesaplamak için aynı müşteriyle tekrar JOIN
JOIN orders o2 ON c.customer_id = o2.customer_id

-- Sadece son 1 yılın verisi alınır
WHERE o.order_date >= DATE '1997-05-06'

GROUP BY
    c.customer_id,
    o.order_id,
    o.order_date,
    c.country,
    c.city

ORDER BY
    c.customer_id,
    o.order_date;

""")
            result = conn.execute(query)
            return result.fetchall()

    def get_query2(self):
        with self.engine.connect() as conn:
            query = text("""WITH product_stats AS (
    SELECT 
        od.product_id,
        AVG(od.discount) AS avg_discount_by_product,
        AVG(od.unit_price * od.quantity * (1 - od.discount)) AS avg_net_spent_by_product
    FROM order_details od
    GROUP BY od.product_id
)

SELECT 
    o.order_id,
    o.customer_id,
    od.product_id,
    p.product_name,
    od.quantity,
    od.unit_price,
    od.discount,
    (od.unit_price * od.quantity * (1 - od.discount)) AS net_spent,
    ps.avg_discount_by_product,
    ps.avg_net_spent_by_product,
    CASE 
        WHEN od.discount >= 0.2 AND (od.unit_price * od.quantity * (1 - od.discount)) < 200 
        THEN 1 
        ELSE 0 
    END AS is_returned
FROM orders o
JOIN order_details od ON o.order_id = od.order_id
JOIN products p ON od.product_id = p.product_id
JOIN product_stats ps ON od.product_id = ps.product_id
WHERE od.quantity > 0
ORDER BY is_returned DESC;
""")
            result = conn.execute(query)
            return result.fetchall()

    def get_query3(self):
        with self.engine.connect() as conn:
            query = text("""WITH musteri_genel_ozet AS (
    SELECT
        c.customer_id,
        COUNT(DISTINCT o.order_id) AS toplam_siparis,
        SUM(od.unit_price * od.quantity * (1 - od.discount)) AS toplam_harcama,
        AVG(od.unit_price) AS ortalama_harcama,
        MAX(o.order_date) AS son_siparis_tarihi
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    JOIN order_details od ON od.order_id = o.order_id
    GROUP BY c.customer_id
),

urun_populerligi AS (
    SELECT
        p.product_id,
        COUNT(DISTINCT od.order_id) AS toplam_siparis_sayisi,
        COUNT(DISTINCT o.customer_id) AS satin_alan_musteri_sayisi
    FROM products p
    LEFT JOIN order_details od ON od.product_id = p.product_id
    LEFT JOIN orders o ON o.order_id = od.order_id
    GROUP BY p.product_id
),

kategori_bazli_musteri_ozet AS (
    SELECT
        c.customer_id,
        cat.category_id,
        SUM(od.unit_price * od.quantity * (1 - od.discount)) AS kategori_harcama,
        SUM(od.quantity) AS kategori_miktar,
        MAX(o.order_date) AS kategori_son_siparis
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    JOIN order_details od ON od.order_id = o.order_id
    JOIN products p ON od.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
    GROUP BY c.customer_id, cat.category_id
),

satin_alinanlar AS (
    SELECT DISTINCT
        o.customer_id,
        od.product_id
    FROM orders o
    JOIN order_details od ON od.order_id = o.order_id
)

SELECT
    c.customer_id AS musteri_id,
    p.product_id AS urun_id,
    cat.category_id AS urun_kategori_id,
    cat.category_name AS urun_kategori_adi,
    p.unit_price AS urun_fiyati,
    p.discontinued AS urun_durumu,

    mgo.toplam_siparis,
    mgo.toplam_harcama,
    mgo.ortalama_harcama,
    mgo.son_siparis_tarihi,
    
    COALESCE(kbo.kategori_harcama, 0) AS bu_kategorideki_harcama,
    COALESCE(kbo.kategori_miktar, 0) AS bu_kategorideki_miktar,
    kbo.kategori_son_siparis,
    
    ROUND((COALESCE(kbo.kategori_harcama, 0) / NULLIF(mgo.toplam_harcama, 0))::numeric, 4) AS kategori_harcama_orani,

    up.toplam_siparis_sayisi AS urun_populerlik_siparis,
    up.satin_alan_musteri_sayisi AS urun_populerlik_musteri,

    ROUND((p.unit_price / NULLIF(mgo.ortalama_harcama, 0))::numeric, 2) AS fiyat_orani_musteriye_gore,

    CASE WHEN sa.product_id IS NOT NULL THEN 1 ELSE 0 END AS satin_alindi_mi

FROM customers c
CROSS JOIN products p
JOIN categories cat ON p.category_id = cat.category_id
LEFT JOIN musteri_genel_ozet mgo ON c.customer_id = mgo.customer_id
LEFT JOIN kategori_bazli_musteri_ozet kbo 
    ON c.customer_id = kbo.customer_id AND kbo.category_id = p.category_id
LEFT JOIN urun_populerligi up ON p.product_id = up.product_id
LEFT JOIN satin_alinanlar sa ON c.customer_id = sa.customer_id AND p.product_id = sa.product_id
ORDER BY c.customer_id, p.product_id;
""")
            result = conn.execute(query)
            return result.fetchall()

    def close(self):
        self.engine.dispose()
