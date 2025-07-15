"""
Enhanced database.py - Simplified version
Only essential improvements while maintaining all original functionality
"""
"""
Fixed database.py - Handle DateTime as index properly
"""
import sqlite3
import pandas as pd
from typing import Union, Optional, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RBAThetaDB:
    """Enhanced SQLite database handler for RBATheta model"""
    
    def __init__(self, db_path: str = ":memory:"):
        """Initialize the database connection"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
        
    def _create_tables(self):
        """Create tables with required columns"""
        # Wind data table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS wind_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            DateTime DATETIME NOT NULL,
            turbine_id TEXT NOT NULL,
            value REAL NOT NULL,
            normalized_value REAL
        )
        """)
        
        # Events table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turbine_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            t1 INTEGER NOT NULL,
            t2 INTEGER NOT NULL,
            delta_t REAL NOT NULL,
            w_t1 REAL NOT NULL,
            w_t2 REAL NOT NULL,
            delta_w REAL NOT NULL,
            sigma REAL NOT NULL,
            theta REAL,
            threshold REAL NOT NULL
        )
        """)
        self.conn.commit()
    
    def load_data(self, data: Union[pd.DataFrame, str], turbine_prefix: str = "Turbine_"):
        """Load data into database - handle DateTime as index or column"""
        if isinstance(data, str):
            data = pd.read_excel(data)
        
        # Make a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Handle DateTime as index or column
        if 'DateTime' not in data_copy.columns:
            if isinstance(data_copy.index, pd.DatetimeIndex):
                # DateTime is the index, reset to make it a column
                data_copy = data_copy.reset_index()
                if data_copy.columns[0] != 'DateTime':
                    data_copy = data_copy.rename(columns={data_copy.columns[0]: 'DateTime'})
            else:
                raise ValueError("DateTime column or DatetimeIndex required")
        
        # Ensure DateTime is datetime type
        data_copy['DateTime'] = pd.to_datetime(data_copy['DateTime'])
        
        # Clear existing data
        self.cursor.execute("DELETE FROM wind_data")
        self.conn.commit()
        
        # Get turbine columns
        turbine_cols = [col for col in data_copy.columns if col.startswith(turbine_prefix)]
        
        if not turbine_cols:
            raise ValueError(f"No columns found with prefix '{turbine_prefix}'")
        
        # Reshape to long format
        data_long = data_copy.melt(
            id_vars='DateTime',
            value_vars=turbine_cols,
            var_name='turbine_id',
            value_name='value'
        )
        
        # Remove any NaN values
        data_long = data_long.dropna()
        
        # Insert data
        data_long[['DateTime', 'turbine_id', 'value']].to_sql(
            'wind_data',
            self.conn,
            if_exists='append',
            index=False
        )
        
        logger.info(f"Loaded {len(data_long)} data points for {len(turbine_cols)} turbines")
    
    def normalize_data(self, nominal: float):
        """Normalize all data using nominal value"""
        # Add normalized_value column if not exists
        try:
            self.cursor.execute("ALTER TABLE wind_data ADD COLUMN normalized_value REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Normalize data
        self.cursor.execute("""
        UPDATE wind_data 
        SET normalized_value = value / ?
        WHERE value IS NOT NULL
        """, (nominal,))
        
        self.conn.commit()
        logger.info(f"Normalized data with nominal value: {nominal}")
    
    def get_turbine_data(self, turbine_id: str) -> pd.DataFrame:
        """Get normalized data for specific turbine"""
        query = """
        SELECT DateTime, normalized_value 
        FROM wind_data 
        WHERE turbine_id = ? AND normalized_value IS NOT NULL
        ORDER BY DateTime
        """
        
        df = pd.read_sql(query, self.conn, params=(turbine_id,), parse_dates=['DateTime'])
        if df.empty:
            logger.warning(f"No data found for turbine {turbine_id}")
            return pd.DataFrame(columns=['normalized_value']).set_index(pd.DatetimeIndex([], name='DateTime'))
        
        return df.set_index('DateTime')
    
    def get_all_turbine_ids(self) -> List[str]:
        """Get all turbine IDs"""
        query = "SELECT DISTINCT turbine_id FROM wind_data WHERE turbine_id IS NOT NULL"
        result = self.cursor.execute(query).fetchall()
        turbine_ids = [x[0] for x in result]
        logger.info(f"Found turbine IDs: {turbine_ids}")
        return turbine_ids
    
    def save_events(self, events: Dict[str, pd.DataFrame], event_type: str):
        """Save events to database"""
        total_events = 0
        
        for turbine_id, df in events.items():
            if df.empty:
                continue
                
            records = []
            for _, row in df.iterrows():
                # Handle different column names flexibly
                delta_t = row.get('∆t_m', row.get('∆t_s', row.get('delta_t', 0)))
                w_t1 = row.get('w_m(t1)', row.get('w_t1', 0))
                w_t2 = row.get('w_m(t2)', row.get('w_t2', 0))
                delta_w = row.get('∆w_m', row.get('delta_w', 0))
                sigma = row.get('σ_m', row.get('σ_s', row.get('sigma', 0)))
                theta = row.get('θ_m', row.get('theta', None))
                
                record = (
                    turbine_id, event_type,
                    int(row.get('t1', 0)), int(row.get('t2', 0)),
                    float(delta_t), float(w_t1), float(w_t2),
                    float(delta_w), float(sigma),
                    float(theta) if pd.notna(theta) else None,
                    float(row.get('threshold', 0))
                )
                records.append(record)
            
            # Save to database
            if records:
                self.cursor.executemany("""
                INSERT INTO events (
                    turbine_id, event_type, t1, t2, delta_t,
                    w_t1, w_t2, delta_w, sigma, theta, threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                total_events += len(records)
        
        self.conn.commit()
        logger.info(f"Saved {total_events} events")
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()