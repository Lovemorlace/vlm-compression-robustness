-- =============================================================================
-- init_schema.sql — Prompt 5.1
-- =============================================================================
-- Schéma complet PostgreSQL pour le projet Compression & VLMs
--
-- Tables :
--   1. images          — Images originales DocLayNet
--   2. ground_truth    — Texte GT reconstruit depuis les JSONs COCO
--   3. compressions    — Images compressées (JPEG + Neural)
--   4. predictions     — Textes prédits par les VLMs
--   5. metrics         — Scores CER/WER/BLEU calculés
--
-- Usage :
--   psql -U vlm_user -d vlm_compression -f platform/database/init_schema.sql
--
-- Ou depuis Python :
--   psql "postgresql://vlm_user:changeme@localhost:5432/vlm_compression" \
--        -f platform/database/init_schema.sql
-- =============================================================================

BEGIN;

-- =============================================================================
-- EXTENSIONS
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- Pour recherche texte approchée si besoin


-- =============================================================================
-- TABLE 1 : images
-- =============================================================================
-- Contient les métadonnées de chaque image originale PNG de DocLayNet.
-- Peuplée par : compress_jpeg.py / compress_neural.py (Phase 2)
-- =============================================================================

CREATE TABLE IF NOT EXISTS images (
    id                  SERIAL PRIMARY KEY,
    image_id            INTEGER NOT NULL UNIQUE,
    filename            VARCHAR(512) NOT NULL,
    category            VARCHAR(128) DEFAULT 'Unknown',
    split               VARCHAR(32)  DEFAULT 'unknown',
    width               INTEGER,
    height              INTEGER,
    original_size_kb    REAL,
    original_path       TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE  images IS 'Images originales DocLayNet (PNG)';
COMMENT ON COLUMN images.image_id IS 'ID unique de l''image dans DocLayNet (depuis le JSON COCO)';
COMMENT ON COLUMN images.category IS 'Catégorie documentaire : Financial, Scientific, Patent, Law, Government, Manual, Unknown';
COMMENT ON COLUMN images.split IS 'Split du dataset : train, val, test';
COMMENT ON COLUMN images.original_size_kb IS 'Taille du fichier PNG original en Ko';
COMMENT ON COLUMN images.original_path IS 'Chemin absolu vers le fichier PNG original';

-- Index
CREATE INDEX IF NOT EXISTS ix_images_image_id   ON images (image_id);
CREATE INDEX IF NOT EXISTS ix_images_category    ON images (category);
CREATE INDEX IF NOT EXISTS ix_images_split       ON images (split);
CREATE INDEX IF NOT EXISTS ix_images_cat_split   ON images (category, split);


-- =============================================================================
-- TABLE 2 : ground_truth
-- =============================================================================
-- Texte ground-truth reconstruit par ordre spatial depuis les annotations
-- COCO JSON de DocLayNet.
-- Peuplée par : prepare_ground_truth.py (Phase 1) puis import CSV,
--               ou directement par un script d'insertion.
-- =============================================================================

CREATE TABLE IF NOT EXISTS ground_truth (
    id                      SERIAL PRIMARY KEY,
    image_id                INTEGER NOT NULL UNIQUE REFERENCES images(image_id),
    gt_text                 TEXT,
    gt_text_normalized      TEXT,
    num_annotations         INTEGER DEFAULT 0,
    num_text_annotations    INTEGER DEFAULT 0,
    num_characters          INTEGER DEFAULT 0,
    num_words               INTEGER DEFAULT 0,
    layout_types            TEXT,
    created_at              TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE  ground_truth IS 'Texte ground-truth reconstruit depuis les annotations COCO DocLayNet';
COMMENT ON COLUMN ground_truth.gt_text IS 'Texte brut reconstruit par tri spatial des bounding boxes';
COMMENT ON COLUMN ground_truth.gt_text_normalized IS 'Texte normalisé (lowercase, espaces nettoyés) utilisé pour le calcul des métriques';
COMMENT ON COLUMN ground_truth.num_annotations IS 'Nombre total d''annotations COCO pour cette image';
COMMENT ON COLUMN ground_truth.num_text_annotations IS 'Nombre d''annotations contenant du texte';
COMMENT ON COLUMN ground_truth.layout_types IS 'Types de layout présents (séparés par |) : Title, Text, Table, etc.';

-- Index
CREATE INDEX IF NOT EXISTS ix_gt_image_id    ON ground_truth (image_id);
CREATE INDEX IF NOT EXISTS ix_gt_num_chars   ON ground_truth (num_characters);


-- =============================================================================
-- TABLE 3 : compressions
-- =============================================================================
-- Chaque ligne = une version compressée d'une image.
-- Une image originale peut avoir N lignes (5 JPEG × QF + 3 Neural × quality).
-- Peuplée par : compress_jpeg.py et compress_neural.py (Phase 2)
-- =============================================================================

CREATE TABLE IF NOT EXISTS compressions (
    id                  SERIAL PRIMARY KEY,
    image_id            INTEGER NOT NULL REFERENCES images(image_id),
    compression_type    VARCHAR(32) NOT NULL,
    compression_level   INTEGER,
    quality_label       VARCHAR(32),
    bitrate_bpp         REAL,
    file_size_kb        REAL,
    compression_ratio   REAL,
    ssim                REAL,
    compressed_path     TEXT,
    created_at          TIMESTAMP DEFAULT NOW(),

    -- Une seule compression par (image, type, niveau)
    CONSTRAINT uq_image_compression
        UNIQUE (image_id, compression_type, compression_level)
);

COMMENT ON TABLE  compressions IS 'Images compressées (JPEG et Neural)';
COMMENT ON COLUMN compressions.compression_type IS 'Type de compression : jpeg ou neural';
COMMENT ON COLUMN compressions.compression_level IS 'Niveau : QF (10-90) pour JPEG, quality (1-6) pour Neural';
COMMENT ON COLUMN compressions.quality_label IS 'Label lisible : QF90, QF70, neural_q1, neural_q3, etc.';
COMMENT ON COLUMN compressions.bitrate_bpp IS 'Bitrate réel en bits per pixel (calculé depuis la taille du bitstream)';
COMMENT ON COLUMN compressions.file_size_kb IS 'Taille du fichier compressé (JPEG) ou du bitstream (Neural) en Ko';
COMMENT ON COLUMN compressions.compression_ratio IS 'Ratio = taille_originale / taille_compressée';
COMMENT ON COLUMN compressions.ssim IS 'SSIM entre l''image originale et compressée (0-1)';

-- Index
CREATE INDEX IF NOT EXISTS ix_comp_image_id      ON compressions (image_id);
CREATE INDEX IF NOT EXISTS ix_comp_type_level     ON compressions (compression_type, compression_level);
CREATE INDEX IF NOT EXISTS ix_comp_bitrate        ON compressions (bitrate_bpp);
CREATE INDEX IF NOT EXISTS ix_comp_type           ON compressions (compression_type);


-- =============================================================================
-- TABLE 4 : predictions
-- =============================================================================
-- Chaque ligne = le texte prédit par un VLM sur une image (originale ou compressée).
-- Peuplée par : infer_qwen2vl.py et infer_internvl2.py (Phase 3)
-- =============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id                      SERIAL PRIMARY KEY,
    image_id                INTEGER NOT NULL REFERENCES images(image_id),
    compression_id          INTEGER REFERENCES compressions(id),
    vlm_name                VARCHAR(64) NOT NULL,
    compression_type        VARCHAR(32) NOT NULL,
    compression_level       INTEGER,
    prompt_used             TEXT,
    predicted_text          TEXT,
    inference_time_s        REAL,
    num_tokens_generated    INTEGER,
    created_at              TIMESTAMP DEFAULT NOW(),

    -- Une seule prédiction par (image, VLM, type compression, niveau)
    CONSTRAINT uq_prediction
        UNIQUE (image_id, vlm_name, compression_type, compression_level)
);

COMMENT ON TABLE  predictions IS 'Textes prédits par les VLMs';
COMMENT ON COLUMN predictions.vlm_name IS 'Nom du VLM : qwen2-vl ou internvl2';
COMMENT ON COLUMN predictions.compression_type IS 'Type : baseline (PNG original), jpeg, ou neural';
COMMENT ON COLUMN predictions.compression_level IS 'Niveau de compression (NULL pour baseline)';
COMMENT ON COLUMN predictions.compression_id IS 'FK vers compressions (NULL pour baseline)';
COMMENT ON COLUMN predictions.prompt_used IS 'Prompt exact envoyé au VLM (pour traçabilité)';
COMMENT ON COLUMN predictions.inference_time_s IS 'Temps d''inférence en secondes';
COMMENT ON COLUMN predictions.num_tokens_generated IS 'Nombre de tokens générés par le VLM';

-- Index
CREATE INDEX IF NOT EXISTS ix_pred_image_id      ON predictions (image_id);
CREATE INDEX IF NOT EXISTS ix_pred_vlm            ON predictions (vlm_name);
CREATE INDEX IF NOT EXISTS ix_pred_vlm_comp       ON predictions (vlm_name, compression_type, compression_level);
CREATE INDEX IF NOT EXISTS ix_pred_comp_id        ON predictions (compression_id);
CREATE INDEX IF NOT EXISTS ix_pred_comp_type      ON predictions (compression_type);


-- =============================================================================
-- TABLE 5 : metrics
-- =============================================================================
-- Chaque ligne = les scores CER/WER/BLEU pour une prédiction.
-- Relation 1:1 avec predictions.
-- Peuplée par : compute_metrics.py (Phase 4)
-- =============================================================================

CREATE TABLE IF NOT EXISTS metrics (
    id                  SERIAL PRIMARY KEY,
    prediction_id       INTEGER NOT NULL UNIQUE REFERENCES predictions(id),
    image_id            INTEGER NOT NULL REFERENCES images(image_id),
    vlm_name            VARCHAR(64) NOT NULL,
    compression_type    VARCHAR(32) NOT NULL,
    compression_level   INTEGER,
    category            VARCHAR(128),
    cer                 REAL,
    wer                 REAL,
    bleu                REAL,
    gt_length           INTEGER,
    pred_length         INTEGER,
    length_ratio        REAL,
    created_at          TIMESTAMP DEFAULT NOW(),

    -- Une seule métrique par prédiction
    CONSTRAINT uq_metric_prediction
        UNIQUE (prediction_id)
);

COMMENT ON TABLE  metrics IS 'Scores de transcription calculés pour chaque prédiction';
COMMENT ON COLUMN metrics.cer IS 'Character Error Rate (0 = parfait, >1 = plus d''erreurs que de caractères)';
COMMENT ON COLUMN metrics.wer IS 'Word Error Rate (0 = parfait)';
COMMENT ON COLUMN metrics.bleu IS 'BLEU score normalisé 0-1 (1 = parfait)';
COMMENT ON COLUMN metrics.gt_length IS 'Nombre de caractères du ground-truth normalisé';
COMMENT ON COLUMN metrics.pred_length IS 'Nombre de caractères du texte prédit normalisé';
COMMENT ON COLUMN metrics.length_ratio IS 'Ratio pred_length / gt_length';
COMMENT ON COLUMN metrics.category IS 'Catégorie documentaire (dénormalisé pour requêtes rapides)';

-- Index pour les requêtes de la plateforme
CREATE INDEX IF NOT EXISTS ix_metrics_pred_id        ON metrics (prediction_id);
CREATE INDEX IF NOT EXISTS ix_metrics_image_id       ON metrics (image_id);
CREATE INDEX IF NOT EXISTS ix_metrics_vlm            ON metrics (vlm_name);
CREATE INDEX IF NOT EXISTS ix_metrics_vlm_comp       ON metrics (vlm_name, compression_type, compression_level);
CREATE INDEX IF NOT EXISTS ix_metrics_category       ON metrics (category);
CREATE INDEX IF NOT EXISTS ix_metrics_comp_type      ON metrics (compression_type);
CREATE INDEX IF NOT EXISTS ix_metrics_cat_vlm_comp   ON metrics (category, vlm_name, compression_type);

-- Index pour les tris et filtres fréquents de la plateforme
CREATE INDEX IF NOT EXISTS ix_metrics_cer            ON metrics (cer);
CREATE INDEX IF NOT EXISTS ix_metrics_wer            ON metrics (wer);
CREATE INDEX IF NOT EXISTS ix_metrics_bleu           ON metrics (bleu);


-- =============================================================================
-- VUES UTILITAIRES — Requêtes préconstruites pour la plateforme
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Vue : résultats complets par image (jointure de tout)
-- Utilisée par l'écran "Vue par image" de la plateforme
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_results_full AS
SELECT
    m.id                    AS metric_id,
    i.image_id,
    i.filename,
    i.category,
    i.split,
    i.width,
    i.height,
    i.original_size_kb,
    i.original_path,
    gt.gt_text,
    gt.num_characters       AS gt_num_chars,
    gt.layout_types,
    p.vlm_name,
    p.compression_type,
    p.compression_level,
    p.predicted_text,
    p.inference_time_s,
    p.num_tokens_generated,
    c.quality_label,
    c.bitrate_bpp,
    c.file_size_kb          AS compressed_size_kb,
    c.compression_ratio,
    c.ssim,
    c.compressed_path,
    m.cer,
    m.wer,
    m.bleu,
    m.gt_length,
    m.pred_length,
    m.length_ratio
FROM metrics m
JOIN predictions p      ON m.prediction_id = p.id
JOIN images i           ON m.image_id = i.image_id
LEFT JOIN ground_truth gt ON i.image_id = gt.image_id
LEFT JOIN compressions c  ON p.compression_id = c.id;

COMMENT ON VIEW v_results_full IS 'Vue complète joignant toutes les tables — utilisée par la plateforme';

-- ---------------------------------------------------------------------------
-- Vue : métriques agrégées par VLM × condition
-- Utilisée par les courbes de dégradation (Écran 3)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_metrics_by_condition AS
SELECT
    vlm_name,
    compression_type,
    compression_level,
    CASE
        WHEN compression_type = 'baseline' THEN 'baseline'
        WHEN compression_type = 'jpeg'     THEN 'jpeg_QF' || compression_level
        WHEN compression_type = 'neural'   THEN 'neural_q' || compression_level
        ELSE compression_type || '_' || compression_level
    END AS condition_label,
    COUNT(*)                AS n_images,
    ROUND(AVG(cer)::numeric, 6)     AS cer_mean,
    ROUND(STDDEV(cer)::numeric, 6)  AS cer_std,
    ROUND(AVG(wer)::numeric, 6)     AS wer_mean,
    ROUND(STDDEV(wer)::numeric, 6)  AS wer_std,
    ROUND(AVG(bleu)::numeric, 6)    AS bleu_mean,
    ROUND(STDDEV(bleu)::numeric, 6) AS bleu_std
FROM metrics
GROUP BY vlm_name, compression_type, compression_level
ORDER BY vlm_name, compression_type, compression_level;

COMMENT ON VIEW v_metrics_by_condition IS 'Métriques moyennes par VLM et condition de compression';

-- ---------------------------------------------------------------------------
-- Vue : métriques agrégées par VLM × condition × catégorie
-- Utilisée par la heatmap (Écran 3)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_metrics_by_category AS
SELECT
    vlm_name,
    compression_type,
    compression_level,
    CASE
        WHEN compression_type = 'baseline' THEN 'baseline'
        WHEN compression_type = 'jpeg'     THEN 'jpeg_QF' || compression_level
        WHEN compression_type = 'neural'   THEN 'neural_q' || compression_level
        ELSE compression_type || '_' || compression_level
    END AS condition_label,
    category,
    COUNT(*)                AS n_images,
    ROUND(AVG(cer)::numeric, 6)  AS cer_mean,
    ROUND(AVG(wer)::numeric, 6)  AS wer_mean,
    ROUND(AVG(bleu)::numeric, 6) AS bleu_mean
FROM metrics
GROUP BY vlm_name, compression_type, compression_level, category
ORDER BY vlm_name, compression_type, compression_level, category;

COMMENT ON VIEW v_metrics_by_category IS 'Métriques par VLM, condition et catégorie documentaire';

-- ---------------------------------------------------------------------------
-- Vue : comparaison JPEG vs Neural à iso-bitrate
-- Utilisée par le graphique comparatif (Écran 3)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_iso_bitrate AS
SELECT
    m.vlm_name,
    m.compression_type,
    m.compression_level,
    m.category,
    c.bitrate_bpp,
    m.cer,
    m.wer,
    m.bleu
FROM metrics m
JOIN predictions p     ON m.prediction_id = p.id
JOIN compressions c    ON p.compression_id = c.id
WHERE m.compression_type IN ('jpeg', 'neural')
ORDER BY m.vlm_name, c.bitrate_bpp;

COMMENT ON VIEW v_iso_bitrate IS 'Données pour comparaison JPEG vs Neural à bitrate comparable';

-- ---------------------------------------------------------------------------
-- Vue : statistiques globales du projet
-- Utile pour le dashboard de la plateforme
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_project_stats AS
SELECT
    (SELECT COUNT(*) FROM images)                   AS total_images,
    (SELECT COUNT(*) FROM ground_truth)             AS total_ground_truths,
    (SELECT COUNT(*) FROM compressions)             AS total_compressions,
    (SELECT COUNT(*) FROM predictions)              AS total_predictions,
    (SELECT COUNT(*) FROM metrics)                  AS total_metrics,
    (SELECT COUNT(DISTINCT vlm_name) FROM predictions) AS total_vlms,
    (SELECT COUNT(DISTINCT category) FROM images)   AS total_categories,
    (SELECT array_agg(DISTINCT vlm_name) FROM predictions) AS vlm_names,
    (SELECT array_agg(DISTINCT category) FROM images)      AS categories;

COMMENT ON VIEW v_project_stats IS 'Statistiques globales du projet';


-- =============================================================================
-- FONCTIONS UTILITAIRES
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Fonction : récupérer les résultats filtrés pour la plateforme
-- Appelée par l'API FastAPI (Phase 6)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_filtered_results(
    p_vlm_name      VARCHAR DEFAULT NULL,
    p_comp_type     VARCHAR DEFAULT NULL,
    p_comp_level    INTEGER DEFAULT NULL,
    p_category      VARCHAR DEFAULT NULL,
    p_limit         INTEGER DEFAULT 100,
    p_offset        INTEGER DEFAULT 0
)
RETURNS TABLE (
    image_id            INTEGER,
    filename            VARCHAR,
    category            VARCHAR,
    vlm_name            VARCHAR,
    compression_type    VARCHAR,
    compression_level   INTEGER,
    condition_label     TEXT,
    bitrate_bpp         REAL,
    ssim                REAL,
    cer                 REAL,
    wer                 REAL,
    bleu                REAL,
    gt_text             TEXT,
    predicted_text      TEXT,
    original_path       TEXT,
    compressed_path     TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.image_id,
        r.filename,
        r.category,
        r.vlm_name,
        r.compression_type,
        r.compression_level,
        CASE
            WHEN r.compression_type = 'baseline' THEN 'baseline'::TEXT
            WHEN r.compression_type = 'jpeg'     THEN ('jpeg_QF' || r.compression_level)::TEXT
            WHEN r.compression_type = 'neural'   THEN ('neural_q' || r.compression_level)::TEXT
            ELSE (r.compression_type || '_' || r.compression_level)::TEXT
        END,
        r.bitrate_bpp,
        r.ssim,
        r.cer,
        r.wer,
        r.bleu,
        r.gt_text,
        r.predicted_text,
        r.original_path,
        r.compressed_path
    FROM v_results_full r
    WHERE (p_vlm_name IS NULL   OR r.vlm_name = p_vlm_name)
      AND (p_comp_type IS NULL  OR r.compression_type = p_comp_type)
      AND (p_comp_level IS NULL OR r.compression_level = p_comp_level)
      AND (p_category IS NULL   OR r.category = p_category)
    ORDER BY r.image_id, r.vlm_name, r.compression_type, r.compression_level
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_filtered_results IS 'Requête paramétrée pour l''API FastAPI — filtres plateforme';


-- =============================================================================
-- SCRIPT D'IMPORT DU GROUND-TRUTH DEPUIS CSV
-- =============================================================================
-- À exécuter après prepare_ground_truth.py (Phase 1.2)
-- Charge le CSV dans la table ground_truth via une table temporaire.
--
-- Usage :
--   1. D'abord, s'assurer que la table images est peuplée
--   2. Puis :
--      psql -U vlm_user -d vlm_compression -c "
--        \COPY tmp_gt(image_id, filename, split, doc_category, gt_text,
--                     num_annotations, num_text_annotations, num_characters,
--                     width, height, layout_types)
--        FROM 'data/metadata/ground_truth.csv' CSV HEADER;
--      "
--   3. Puis exécuter import_ground_truth_from_tmp() ci-dessous
-- =============================================================================

-- Table temporaire pour l'import CSV
CREATE TABLE IF NOT EXISTS tmp_gt (
    image_id                INTEGER,
    filename                VARCHAR(512),
    split                   VARCHAR(32),
    doc_category            VARCHAR(128),
    gt_text                 TEXT,
    num_annotations         INTEGER,
    num_text_annotations    INTEGER,
    num_characters          INTEGER,
    width                   INTEGER,
    height                  INTEGER,
    layout_types            TEXT
);

-- Fonction d'import depuis tmp_gt vers ground_truth
CREATE OR REPLACE FUNCTION import_ground_truth_from_tmp()
RETURNS void AS $$
BEGIN
    -- Insérer dans images si manquant
    INSERT INTO images (image_id, filename, category, split, width, height)
    SELECT image_id, filename, doc_category, split, width, height
    FROM tmp_gt
    ON CONFLICT (image_id) DO NOTHING;

    -- Insérer dans ground_truth
    INSERT INTO ground_truth (
        image_id, gt_text, gt_text_normalized,
        num_annotations, num_text_annotations,
        num_characters, num_words, layout_types
    )
    SELECT
        image_id,
        gt_text,
        LOWER(REGEXP_REPLACE(TRIM(gt_text), '\s+', ' ', 'g')),
        num_annotations,
        num_text_annotations,
        num_characters,
        array_length(string_to_array(TRIM(gt_text), ' '), 1),
        layout_types
    FROM tmp_gt
    WHERE gt_text IS NOT NULL AND gt_text != ''
    ON CONFLICT (image_id) DO UPDATE SET
        gt_text = EXCLUDED.gt_text,
        gt_text_normalized = EXCLUDED.gt_text_normalized,
        num_annotations = EXCLUDED.num_annotations,
        num_text_annotations = EXCLUDED.num_text_annotations,
        num_characters = EXCLUDED.num_characters,
        num_words = EXCLUDED.num_words,
        layout_types = EXCLUDED.layout_types;

    -- Nettoyer
    TRUNCATE tmp_gt;

    RAISE NOTICE 'Import ground-truth terminé.';
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- PERMISSIONS
-- =============================================================================
-- Accorder les droits nécessaires (adapter vlm_user si besoin)

DO $$
BEGIN
    -- Tables
    EXECUTE format('GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO %I', 'vlm_user');
    -- Séquences (pour les SERIAL)
    EXECUTE format('GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO %I', 'vlm_user');
    -- Fonctions
    EXECUTE format('GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO %I', 'vlm_user');
EXCEPTION
    WHEN undefined_object THEN
        RAISE NOTICE 'Utilisateur vlm_user non trouvé — permissions ignorées';
END;
$$;


-- =============================================================================
-- VÉRIFICATION FINALE
-- =============================================================================
DO $$
DECLARE
    t_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO t_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE';

    RAISE NOTICE '=== Initialisation terminée ===';
    RAISE NOTICE 'Tables créées : % (attendu : 6 avec tmp_gt)', t_count;
    RAISE NOTICE 'Vues créées   : v_results_full, v_metrics_by_condition, v_metrics_by_category, v_iso_bitrate, v_project_stats';
    RAISE NOTICE 'Fonctions     : get_filtered_results(), import_ground_truth_from_tmp()';
    RAISE NOTICE '================================';
END;
$$;

COMMIT;
