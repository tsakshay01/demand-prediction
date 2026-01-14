require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

// Config
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY || SUPABASE_URL === 'YOUR_SUPABASE_URL') {
    console.warn("⚠️  SUPABASE CONFIG MISSING. Running in MOCK MODE (In-memory).");
    console.warn("   Details: Add SUPABASE_URL and SUPABASE_KEY to .env");
}

const supabase = (SUPABASE_URL && SUPABASE_KEY && SUPABASE_URL !== 'YOUR_SUPABASE_URL')
    ? createClient(SUPABASE_URL, SUPABASE_KEY)
    : null;

module.exports = {
    supabase,
    isMock: !supabase
};
