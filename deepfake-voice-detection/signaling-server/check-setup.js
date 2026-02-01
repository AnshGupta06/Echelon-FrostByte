// Quick diagnostic script to check if database is set up correctly
require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
  user: process.env.DB_USER || 'postgres',
  host: process.env.DB_HOST || 'localhost',
  database: process.env.DB_NAME || 'deepfake_calls',
  password: process.env.DB_PASSWORD || 'postgres',
  port: process.env.DB_PORT || 5432,
});

async function checkSetup() {
  console.log('🔍 Checking database setup...\n');

  try {
    // Test connection
    console.log('1. Testing database connection...');
    await pool.query('SELECT NOW()');
    console.log('   ✅ Database connection successful\n');

    // Check if tables exist
    console.log('2. Checking if tables exist...');
    const tablesResult = await pool.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY table_name
    `);

    const expectedTables = ['users', 'calls', 'audio_chunks', 'audio_chunk_predictions'];
    const existingTables = tablesResult.rows.map(r => r.table_name);

    expectedTables.forEach(table => {
      if (existingTables.includes(table)) {
        console.log(`   ✅ Table '${table}' exists`);
      } else {
        console.log(`   ❌ Table '${table}' is missing`);
      }
    });

    if (existingTables.length === 0) {
      console.log('\n   ⚠️  No tables found! Run: psql -d deepfake_calls -f database.sql\n');
    } else {
      console.log('\n   ✅ All required tables exist\n');
    }

    // Check if database is empty
    const userCount = await pool.query('SELECT COUNT(*) FROM users');
    console.log(`3. Current users in database: ${userCount.rows[0].count}\n`);

    console.log('✅ Database setup looks good!');
    process.exit(0);

  } catch (error) {
    console.error('\n❌ Database setup check failed:\n');
    
    if (error.code === 'ECONNREFUSED') {
      console.error('   Error: Cannot connect to PostgreSQL');
      console.error('   Solution: Make sure PostgreSQL is running');
      console.error('   Command: Get-Service -Name postgresql*');
    } else if (error.code === '3D000') {
      console.error(`   Error: Database '${process.env.DB_NAME || 'deepfake_calls'}' does not exist`);
      console.error('   Solution: Create the database first');
      console.error('   Command: createdb deepfake_calls');
    } else if (error.code === '28P01') {
      console.error('   Error: Authentication failed');
      console.error('   Solution: Check your DB_USER and DB_PASSWORD in .env file');
    } else {
      console.error(`   Error: ${error.message}`);
      console.error(`   Code: ${error.code}`);
    }
    
    console.error('\n📝 See TROUBLESHOOTING.md for more help\n');
    process.exit(1);
  } finally {
    await pool.end();
  }
}

checkSetup();

