-- Run this in your Supabase SQL Editor

-- Users Table
create table users (
  id uuid default uuid_generate_v4() primary key,
  email text unique not null,
  password_hash text not null,
  name text,
  role text default 'user',
  settings jsonb default '{"theme": "dark", "notifications": false}',
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Files Table
create table files (
  id uuid default uuid_generate_v4() primary key,
  user_id uuid references users(id),
  filename text not null,
  original_name text not null,
  size bigint,
  status text default 'Processed',
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Seed Admin User (Password: 'admin')
-- Note: In production application, you would hash this password manually or via script.
-- For this setup, the server will hash it on first run or we can insert a dummy hash.
-- This SQL just creates the structure. The application will handle insertion.
