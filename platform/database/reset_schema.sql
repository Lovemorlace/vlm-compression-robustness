-- =============================================================================
-- reset_schema.sql
-- =============================================================================
-- Supprime toutes les tables, vues et fonctions pour repartir de zéro.
-- ATTENTION : PERTE DE DONNÉES IRRÉVERSIBLE
--
-- Usage :
--   psql -U vlm_user -d vlm_compression -f platform/database/reset_schema.sql
--   psql -U vlm_user -d vlm_compression -f platform/database/init_schema.sql
-- =============================================================================

BEGIN;

-- Supprimer les vues
DROP VIEW IF EXISTS v_project_stats CASCADE;
DROP VIEW IF EXISTS v_iso_bitrate CASCADE;
DROP VIEW IF EXISTS v_metrics_by_category CASCADE;
DROP VIEW IF EXISTS v_metrics_by_condition CASCADE;
DROP VIEW IF EXISTS v_results_full CASCADE;

-- Supprimer les fonctions
DROP FUNCTION IF EXISTS get_filtered_results CASCADE;
DROP FUNCTION IF EXISTS import_ground_truth_from_tmp CASCADE;

-- Supprimer les tables (ordre : dépendances d'abord)
DROP TABLE IF EXISTS metrics CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS compressions CASCADE;
DROP TABLE IF EXISTS ground_truth CASCADE;
DROP TABLE IF EXISTS tmp_gt CASCADE;
DROP TABLE IF EXISTS images CASCADE;

COMMIT;

-- Vérification
DO $$
DECLARE
    t_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO t_count
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    RAISE NOTICE 'Reset terminé. Tables restantes : %', t_count;
END;
$$;
