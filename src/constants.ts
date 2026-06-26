/**
 * Sample chest X-ray used when "Run Analysis" is clicked without an uploaded
 * image. Self-hosted in public/ (base-path aware) so the demo no longer depends
 * on an external CDN (upload.wikimedia.org was slow/blocked in some regions).
 *
 * Source: Wikimedia Commons — "Chest X-ray in influenza and Haemophilus
 * influenzae" (annotated):
 * https://commons.wikimedia.org/wiki/File:Chest_X-ray_in_influenza_and_Haemophilus_influenzae_-_annotated.jpg
 */
export const SAMPLE_IMAGE_URL = `${import.meta.env.BASE_URL}sample-xray.jpg`;
